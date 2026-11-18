"""Bracket short MyoFullBody skill clips with the guard pose so they can be chained.

Each clip gets, when its first or last frame is away from guard:

    [guard hold] [blend guard -> first frame] [original clip] [blend last frame -> guard] [guard hold]

The guard pose is taken from a reference clip (``guard.npz``). It is placed at the
clip's own root position [m] and heading [rad], so a chain of bracketed clips never
teleports the body: only the posture returns to guard. Joint angles are blended
with a smoothstep, the root orientation with a slerp, and the root height is
corrected after the blend so the lowest foot site follows the blended foot height
(no floor penetration or floating during the transition).

All per-frame arrays (qvel, xpos, xquat, cvel, site_xpos, site_xmat, subtree_com)
are recomputed from qpos with MuJoCo forward kinematics. The musclemimic loader
recomputes qvel and the kinematic caches from qpos anyway; filling them keeps the
files consistent for other tools.

Clips with joint angles outside the model's joint limits are refused, because a
blend into a physically impossible pose would be meaningless.

Usage::

    python scripts/bracket_clips_with_guard.py \
        --src_dir <...>/clips/MyoFullBody/gmr_v6/skills \
        --dst_dir <...>/guard_clips/MyoFullBody/gmr_v6/skills
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FREE_JOINT_QPOS = 7  # root free joint: xyz [m] + quaternion wxyz
PER_FRAME_KEYS = ("qpos", "qvel", "xpos", "xquat", "cvel", "site_xpos", "site_xmat", "subtree_com")
FOOT_SITES = ("right_ankle_mimic", "right_toes_mimic", "left_ankle_mimic", "left_toes_mimic")
# Pose distance [m] below which a clip end already counts as guard (mean site error).
GUARD_TOLERANCE_M = 0.03
# Joint-limit violation [rad] above which a clip is refused (about 5 deg).
JOINT_LIMIT_TOLERANCE_RAD = np.radians(5.0)


def load_model() -> mujoco.MjModel:
    """Compile the MyoFullBody model exactly as the trainer does (fingers disabled)."""
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    return MyoFullBody()._model


class Kinematics:
    """Forward kinematics for one model and one fixed list of tracked sites."""

    def __init__(self, model: mujoco.MjModel, site_names: list[str]) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in site_names])
        if (self.site_ids < 0).any():
            raise ValueError("clip site names do not exist in the model")
        foot = [site_names.index(s) for s in FOOT_SITES]
        self.foot_ids = self.site_ids[foot]

    def forward(self, qpos: np.ndarray, qvel: np.ndarray | None = None) -> mujoco.MjData:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0 if qvel is None else qvel
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)
        return self.data

    def min_foot_z(self, qpos: np.ndarray) -> float:
        return float(self.forward(qpos).site_xpos[self.foot_ids, 2].min())

    def local_sites(self, qpos: np.ndarray) -> np.ndarray:
        """Tracked site positions [m] relative to the root, in the root's heading frame."""
        sites = self.forward(qpos).site_xpos[self.site_ids] - qpos[:3]
        return Rotation.from_euler("z", -heading(qpos)).apply(sites)


def root_rotation(qpos: np.ndarray) -> Rotation:
    w, x, y, z = qpos[3:7]
    return Rotation.from_quat([x, y, z, w])


def set_root_rotation(qpos: np.ndarray, rot: Rotation) -> None:
    x, y, z, w = rot.as_quat()
    qpos[3:7] = [w, x, y, z]


def heading(qpos: np.ndarray) -> float:
    """Root yaw [rad] about the world z axis."""
    return float(root_rotation(qpos).as_euler("ZYX")[0])


def guard_at(guard: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Guard posture placed at the anchor frame's root xy [m] and heading [rad]."""
    tilt = Rotation.from_euler("z", -heading(guard)) * root_rotation(guard)
    q = guard.copy()
    q[:2] = anchor[:2]
    set_root_rotation(q, Rotation.from_euler("z", heading(anchor)) * tilt)
    return q


def smoothstep(s: np.ndarray) -> np.ndarray:
    return s * s * (3.0 - 2.0 * s)


def blend(a: np.ndarray, b: np.ndarray, n: int, kin: Kinematics) -> np.ndarray:
    """``n`` frames strictly between poses ``a`` and ``b`` (both endpoints excluded)."""
    s = smoothstep(np.arange(1, n + 1) / (n + 1))
    frames = a + s[:, None] * (b - a)
    rots = Slerp([0.0, 1.0], Rotation.concatenate([root_rotation(a), root_rotation(b)]))(s)
    foot_a, foot_b = kin.min_foot_z(a), kin.min_foot_z(b)
    for k in range(n):
        set_root_rotation(frames[k], rots[k])
        target = foot_a + s[k] * (foot_b - foot_a)
        frames[k, 2] += target - kin.min_foot_z(frames[k])
    return frames


def blend_frames(distance_m: float, min_s: float, max_s: float, hz: float) -> int:
    """Blend length [frames]: longer for poses further from guard, 0.30 m maps to the max."""
    seconds = min_s + np.clip(distance_m / 0.30, 0.0, 1.0) * (max_s - min_s)
    return int(round(seconds * hz))


def joint_limit_violation(qpos: np.ndarray, joint_names: list[str], model: mujoco.MjModel) -> tuple[float, str]:
    """Largest excursion [rad] beyond any limited hinge/slide joint, and that joint's name."""
    worst, worst_name = 0.0, ""
    for k, name in enumerate(joint_names[1:]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or not model.jnt_limited[jid]:
            continue
        lo, hi = model.jnt_range[jid]
        v = qpos[:, FREE_JOINT_QPOS + k]
        over = float(np.maximum(lo - v, v - hi).max())
        if over > worst:
            worst, worst_name = over, name
    return worst, worst_name


def recompute_arrays(qpos: np.ndarray, hz: float, kin: Kinematics) -> dict[str, np.ndarray]:
    """qvel from centred differences (as the trainer's loader does), the rest from FK."""
    model = kin.model
    n = len(qpos)
    fwd = np.zeros((n - 1, model.nv))
    for t in range(n - 1):
        mujoco.mj_differentiatePos(model, fwd[t], 1.0 / hz, qpos[t], qpos[t + 1])
    qvel = np.zeros((n, model.nv))
    qvel[0], qvel[-1] = fwd[0], fwd[-1]
    qvel[1:-1] = 0.5 * (fwd[:-1] + fwd[1:])

    out = {k: [] for k in PER_FRAME_KEYS}
    for t in range(n):
        d = kin.forward(qpos[t], qvel[t])
        out["xpos"].append(d.xpos.copy())
        out["xquat"].append(d.xquat.copy())
        out["cvel"].append(d.cvel.copy())
        out["subtree_com"].append(d.subtree_com.copy())
        out["site_xpos"].append(d.site_xpos[kin.site_ids].copy())
        out["site_xmat"].append(d.site_xmat[kin.site_ids].copy())
    out["qpos"], out["qvel"] = qpos, qvel
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def bracket_clip(clip: dict, guard: np.ndarray, kin: Kinematics, args: argparse.Namespace) -> tuple[dict, str]:
    hz = float(clip["frequency"])
    qpos = clip["qpos"].astype(np.float64)
    hold = int(round(args.hold_s * hz))
    pieces, notes = [], []
    for end in ("start", "end"):
        frame = qpos[0] if end == "start" else qpos[-1]
        g = guard_at(guard, frame)
        dist = float(np.linalg.norm(kin.local_sites(frame) - kin.local_sites(g), axis=1).mean())
        if dist <= GUARD_TOLERANCE_M:
            pieces.append(np.empty((0, qpos.shape[1])))
            notes.append(f"{end} already guard ({dist * 100:.1f} cm)")
            continue
        n = blend_frames(dist, args.min_blend_s, args.max_blend_s, hz)
        holds = np.repeat(g[None], hold, axis=0)
        if end == "start":
            pieces.append(np.concatenate([holds, blend(g, frame, n, kin)]))
        else:
            pieces.append(np.concatenate([blend(frame, g, n, kin), holds]))
        notes.append(f"{end} {dist * 100:.1f} cm -> {hold}+{n} frames")
    new_qpos = np.concatenate([pieces[0], qpos, pieces[1]])

    out = {k: clip[k] for k in clip if k not in PER_FRAME_KEYS}
    out.update(recompute_arrays(new_qpos, hz, kin))
    out["split_points"] = np.array([0, len(new_qpos)], dtype=np.int32)
    return out, "; ".join(notes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", type=Path, required=True)
    p.add_argument("--dst_dir", type=Path, required=True)
    p.add_argument("--guard", type=Path, default=None, help="guard clip (default: <src_dir>/guard.npz)")
    p.add_argument("--hold_s", type=float, default=0.2, help="guard hold before/after the blend [s]")
    p.add_argument("--min_blend_s", type=float, default=0.3, help="blend length for a pose near guard [s]")
    p.add_argument("--max_blend_s", type=float, default=0.6, help="blend length for a pose 0.30 m from guard [s]")
    args = p.parse_args()

    guard_file = np.load(args.guard or args.src_dir / "guard.npz", allow_pickle=True)
    guard = guard_file["qpos"][0].astype(np.float64)
    model = load_model()
    kin = Kinematics(model, [str(s) for s in guard_file["site_names"]])
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(args.src_dir.glob("*.npz")):
        f = np.load(path, allow_pickle=True)
        clip = {k: f[k] for k in f.files}
        name = path.stem
        if [str(s) for s in clip["site_names"]] != [str(s) for s in guard_file["site_names"]]:
            print(f"SKIP  {name:24s} site list differs from the guard clip")
            continue
        over, joint = joint_limit_violation(clip["qpos"], [str(j) for j in clip["joint_names"]], model)
        if over > JOINT_LIMIT_TOLERANCE_RAD:
            print(f"SKIP  {name:24s} {joint} is {np.degrees(over):.0f} deg past its joint limit (re-retarget it)")
            continue
        out, note = bracket_clip(clip, guard, kin, args)
        np.savez(args.dst_dir / path.name, **out)
        n_old, n_new = len(clip["qpos"]), len(out["qpos"])
        print(f"WROTE {name:24s} {n_old:4d} -> {n_new:4d} frames  ({note})")


if __name__ == "__main__":
    main()
