"""Build the boxing skill library: cut the mazen_c3d mocap clips into short skills.

Four subcommands::

    # suggest punch segments (prints a table and writes a draft manifest)
    python scripts/boxing_skills.py detect --out logs/boxing_skills_draft.yaml

    # cut every skill in the manifest into its own padded .npz clip
    python scripts/boxing_skills.py cut

    # render one review video per skill plus an all-skills reel
    python scripts/boxing_skills.py render

    # loop skills in the interactive MuJoCo viewer (macOS: run with mjpython);
    # --vs plays them on one boxer of the two-boxer BoxingVs scene
    mjpython scripts/boxing_skills.py view punch_R_push1
    mjpython scripts/boxing_skills.py view punch_R_push1 --vs --opponent guard

The manifest (``boxing_skills.yaml`` next to the BoxingVs task) is the source of truth:
edit a frame range there and re-run ``cut``. Every skill starts in the ``guard`` pose
(eased into the mocap over ``lead`` frames) and ends with a generated return to it
(``blend`` frames), so skills can be chained. Segments stay in mocap coordinates;
re-anchoring to the boxer happens at runtime (``anchor_qpos``).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myosuite.envs.myo.tasks.challenge.boxing_vs.skill_library import (
    MANIFEST,
    SKILL_DIR,
    anchor_qpos,
    yaw_quat,
)

REPO = Path(__file__).resolve().parents[1]
CLIP_DIR = REPO / "boxing_motions/MyoFullBody/gmr/Transitions_mocap/mazen_c3d"
VIDEO_DIR = REPO / "logs/videos/boxing_skills"

# Per-frame arrays that are sliced; everything else in the .npz is model metadata.
PER_FRAME = (
    "qpos",
    "qvel",
    "xpos",
    "xquat",
    "cvel",
    "subtree_com",
    "site_xpos",
    "site_xmat",
)
VELOCITY_FIELDS = ("qvel", "cvel")

# Frame windows of each clip where the actor punches in place (read off the signal plots).
PUNCH_WINDOWS: dict[str, tuple[int, int]] = {
    "punchboxing_push_poses": (25, 160),
    "punchboxing_stand_poses": (25, 165),
    "punchboxing_walk_poses": (25, 200),
    "punchboxing_kick_poses": (25, 190),
    "punchboxing_jumpinplace_poses": (25, 215),
    "punchboxing_running_poses": (195, 300),
}


@dataclass
class Skill:
    """One manifest entry: a frame range of a source clip, returned to guard and held."""

    name: str
    clip: str
    start: int
    end: int
    pad: int = 15  # frames held in the final pose
    blend: int = 40  # frames of generated return-to-guard motion (0 = none)
    lead: int = 20  # frames over which the start eases from guard into the mocap
    joints: dict[str, float] = field(default_factory=dict)  # guard pose overrides (rad)
    kind: str = "punch_R"  # menu group: punch_R, punch_L, combo, guard, step_fwd, ...
    note: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> Skill:
        """Build from ``{clip, frames: [start, end], pad, blend, lead, kind, joints, note}``."""
        start, end = d["frames"]
        return cls(
            name=name,
            clip=d["clip"],
            start=int(start),
            end=int(end),
            pad=int(d.get("pad", 15)),
            blend=int(d.get("blend", 40)),
            lead=int(d.get("lead", 20)),
            joints={k: float(v) for k, v in d.get("joints", {}).items()},
            kind=d.get("kind", "punch_R"),
            note=d.get("note", ""),
        )


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _site(d: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    names = [str(s) for s in d["site_names"]]
    return np.asarray(d["site_xpos"][:, names.index(name)], dtype=np.float64)


def hand_signals(d: np.lib.npyio.NpzFile) -> dict[str, dict[str, np.ndarray]]:
    """Smoothed per-hand reach (horizontal, from pelvis), speed and height above the head."""
    hz = float(d["frequency"])
    pelvis, head = _site(d, "pelvis_mimic"), _site(d, "head_mimic")
    out = {}
    for side, site in (("R", "right_hand_mimic"), ("L", "left_hand_mimic")):
        h = _site(d, site)
        speed = np.r_[0.0, np.linalg.norm(np.diff(h, axis=0), axis=1) * hz]
        out[side] = {
            "reach": _smooth(np.linalg.norm((h - pelvis)[:, :2], axis=1)),
            "speed": _smooth(speed),
            "height": h[:, 2] - head[:, 2],
        }
    return out


def detect_punches(
    d: np.lib.npyio.NpzFile,
    window: tuple[int, int],
    min_reach: float = 0.42,
    min_rise: float = 0.12,
    min_speed: float = 2.0,
) -> list[dict]:
    """Find single punches: a hand-reach peak between two minima of the combined reach.

    Args:
        d: Loaded clip.
        window: ``(start, end)`` frames to search.
        min_reach: Minimum horizontal hand reach at the peak (m).
        min_rise: Minimum reach gained from the lower bracketing minimum (m).
        min_speed: Minimum hand speed within 10 frames of the peak (m/s).

    Returns:
        Punches sorted by start frame: ``{hand, start, peak, end, reach, speed, height}``.
    """
    a, b = window
    sig = hand_signals(d)
    # Cut points are minima of the combined reach, so a cut never lands mid-punch of
    # the other hand.
    both = _smooth(sig["R"]["reach"] + sig["L"]["reach"], 7)
    cuts = (
        [a]
        + [
            i
            for i in range(a + 1, b - 1)
            if both[i] <= both[i - 1] and both[i] < both[i + 1]
        ]
        + [b - 1]
    )
    found = []
    for side, s in sig.items():
        r = s["reach"]
        for p in range(a + 1, b - 1):
            if not (r[p] >= r[p - 1] and r[p] > r[p + 1] and r[p] > min_reach):
                continue
            lo = max(c for c in cuts if c < p)
            hi = min(c for c in cuts if c > p)
            speed = float(s["speed"][max(a, p - 10) : p + 10].max())
            if r[p] - min(r[lo], r[hi]) < min_rise or speed < min_speed:
                continue
            if hi == b - 1:  # runs past the window: punch may be cut short
                continue
            found.append(
                {
                    "hand": side,
                    "start": int(lo),
                    "peak": int(p),
                    "end": int(hi),
                    "reach": round(float(r[p]), 2),
                    "speed": round(speed, 1),
                    "height": round(float(s["height"][p]), 2),
                }
            )
    # Two hands peaking between the same cuts is a double punch: keep the faster one.
    best: dict[tuple[int, int], dict] = {}
    for f in found:
        key = (f["start"], f["end"])
        if key not in best or f["speed"] > best[key]["speed"]:
            best[key] = f
    return sorted(best.values(), key=lambda x: x["start"])


def guard_frame(d: np.lib.npyio.NpzFile, window: tuple[int, int]) -> int:
    """Frame in ``window`` with both hands pulled in and the least joint motion."""
    a, b = window
    sig = hand_signals(d)
    joint_speed = _smooth(np.linalg.norm(d["qvel"][:, 6:], axis=1))
    score = (
        sig["R"]["reach"]
        + sig["L"]["reach"]
        + 0.01 * joint_speed
        + sig["R"]["speed"] * 0.05
    )
    return int(a + np.argmin(score[a:b]))


def cmd_detect(args: argparse.Namespace) -> None:
    """Print detected punches per clip and write them as a draft manifest."""
    draft: dict[str, dict] = {}
    for stem, window in PUNCH_WINDOWS.items():
        d = np.load(CLIP_DIR / f"{stem}.npz", allow_pickle=True)
        short = stem.removeprefix("punchboxing_").removesuffix("_poses")
        print(f"\n{stem}  window={window}  guard frame={guard_frame(d, window)}")
        counts = {"R": 0, "L": 0}
        for p in detect_punches(d, window):
            counts[p["hand"]] += 1
            name = f"punch_{p['hand']}_{short}{counts[p['hand']]}"
            print(
                f"  {name:24s} frames {p['start']:4d}-{p['end']:4d}  peak {p['peak']:4d}"
                f"  reach {p['reach']:.2f} m  speed {p['speed']:.1f} m/s  "
                f"hand-head z {p['height']:+.2f} m"
            )
            draft[name] = {
                "clip": stem,
                "frames": [p["start"], p["end"]],
                "kind": f"punch_{p['hand']}",
                "note": f"peak {p['peak']}, reach {p['reach']} m, {p['speed']} m/s",
            }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump({"skills": draft}, sort_keys=False))
        print(f"\nDraft manifest: {args.out}")


def load_manifest(path: Path = MANIFEST) -> list[Skill]:
    """Read the skill manifest."""
    raw = yaml.safe_load(path.read_text())
    return [Skill.from_dict(k, v) for k, v in raw["skills"].items()]


@lru_cache(maxsize=1)
def fullbody_model() -> Any:
    """Compiled MyoFullBody model the clips were retargeted to."""
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    return compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())[0]


def guard_qpos(skills: list[Skill]) -> np.ndarray:
    """Reference guard pose: the ``guard`` skill's frame plus its ``joints`` overrides."""
    guard = next(s for s in skills if s.name == "guard")
    src = np.load(CLIP_DIR / f"{guard.clip}.npz", allow_pickle=True)
    qpos = np.asarray(src["qpos"][guard.start], dtype=np.float64).copy()
    model = fullbody_model()
    for name, value in guard.joints.items():
        qpos[model.joint(name).qposadr[0]] = value
    return qpos


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return 3 * x**2 - 2 * x**3


def _guard_at(q: np.ndarray, q_guard: np.ndarray) -> np.ndarray:
    """Guard pose placed at the root position and heading of ``q``."""
    import mujoco

    out = np.asarray(q_guard, np.float64).copy()
    out[:2] = q[:2]
    yaw_inv, tilt = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(yaw_inv, yaw_quat(out[3:7]))
    mujoco.mju_mulQuat(tilt, yaw_inv, out[3:7])
    mujoco.mju_mulQuat(out[3:7], yaw_quat(np.asarray(q[3:7], np.float64)), tilt)
    return out


def lead_from_guard(qpos: np.ndarray, q_guard: np.ndarray) -> np.ndarray:
    """Ease the start of a motion out of the guard pose.

    The difference between the guard and the first frame is added to the motion and
    faded out, so the move launches from guard instead of first dropping to the
    mocap start pose.

    Args:
        qpos: First frames of the motion ``(n, nq)``.
        q_guard: Guard ``qpos``.

    Returns:
        ``qpos`` whose first frame is the guard pose and whose offset fades to zero
        after the last frame.
    """
    import mujoco

    out = np.asarray(qpos, np.float64).copy()
    start = _guard_at(out[0], q_guard)
    rot = np.zeros(3)
    mujoco.mju_subQuat(rot, start[3:7], out[0, 3:7])
    offset = start - out[0]
    for q, w in zip(out, 1 - _smoothstep(np.arange(len(out)) / len(out))):
        q[2] += w * offset[2]
        q[7:] += w * offset[7:]
        mujoco.mju_quatIntegrate(q[3:7], rot, w)
    return out


def blend_to_guard(
    q_end: np.ndarray, v_end: np.ndarray, q_guard: np.ndarray, n: int, hz: float
) -> np.ndarray:
    """Generate ``n`` frames moving from a clip's last pose to the guard pose.

    Joints and root height follow a cubic that keeps the clip's end velocity and
    arrives at rest. The root keeps its position (coasting to a stop) and heading;
    only its tilt goes to the guard's.

    Args:
        q_end: Last ``qpos`` of the motion.
        v_end: ``qvel`` at that frame.
        q_guard: Guard ``qpos``.
        n: Number of generated frames; the last one is the guard pose.
        hz: Clip frame rate.

    Returns:
        ``qpos`` of shape ``(n, nq)``.
    """
    import mujoco

    model = fullbody_model()
    q0 = np.asarray(q_end, np.float64)
    q1 = _guard_at(q0, q_guard)
    v0 = np.r_[v_end[:3], np.zeros(4), v_end[6:]].astype(np.float64)
    duration = n / hz
    q1[:2] = q0[:2] + v0[:2] * duration / 2  # constant deceleration to rest

    s = (np.arange(1, n + 1) / n)[:, None]
    out = (
        (2 * s**3 - 3 * s**2 + 1) * q0
        + (s**3 - 2 * s**2 + s) * duration * v0
        + _smoothstep(s) * q1
    )
    # Keep the velocity-matched cubic inside the joint limits.
    for j in np.flatnonzero(model.jnt_limited[1:]) + 1:
        a = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        out[:, a] = np.clip(out[:, a], min(lo, q0[a], q1[a]), max(hi, q0[a], q1[a]))
    rot = np.zeros(3)
    mujoco.mju_subQuat(rot, q1[3:7], q0[3:7])
    for i, w in enumerate(_smoothstep(s[:, 0])):
        quat = q0[3:7].copy()
        mujoco.mju_quatIntegrate(quat, rot, w)
        out[i, 3:7] = quat
    return out


def derive_frames(
    path: np.ndarray, hz: float, site_names: list[str]
) -> dict[str, np.ndarray]:
    """Recompute every per-frame clip field from ``qpos`` by forward kinematics.

    Args:
        path: Poses ``(n + 2, nq)``: the frame before, the ``n`` frames to derive
            and the frame after (the neighbours give central-difference velocities).
        hz: Clip frame rate.
        site_names: Clip site order.

    Returns:
        ``PER_FRAME`` arrays of length ``n``, float32.
    """
    import mujoco

    model = fullbody_model()
    data = mujoco.MjData(model)
    site_ids = [model.site(name).id for name in site_names]
    out: dict[str, list[np.ndarray]] = {k: [] for k in PER_FRAME}
    for i in range(1, len(path) - 1):
        data.qpos[:] = path[i]
        mujoco.mj_differentiatePos(model, data.qvel, 2 / hz, path[i - 1], path[i + 1])
        mujoco.mj_forward(model, data)
        frame = {k: getattr(data, k) for k in PER_FRAME if not k.startswith("site_")}
        frame["site_xpos"] = data.site_xpos[site_ids]
        frame["site_xmat"] = data.site_xmat[site_ids]
        for k in PER_FRAME:
            out[k].append(np.array(frame[k], dtype=np.float32))
    return {k: np.stack(v) for k, v in out.items()}


def cut_skill(skill: Skill, guard: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Slice one skill out of its source clip, bracket it with the guard and hold the end.

    Args:
        skill: Manifest entry.
        guard: Guard ``qpos``. The first ``skill.lead`` frames ease out of it and
            ``skill.blend`` generated frames return to it. ``None`` leaves the mocap
            untouched and only holds its last frame.

    Returns:
        Clip arrays: motion, then ``blend`` generated frames, then ``pad`` held frames.
    """
    src = np.load(CLIP_DIR / f"{skill.clip}.npz", allow_pickle=True)
    n = src["qpos"].shape[0]
    if not 0 <= skill.start < skill.end <= n:
        raise ValueError(
            f"{skill.name}: frames {skill.start}-{skill.end} outside 0-{n}"
        )
    out = {k: src[k] for k in src.files}
    hz = float(src["frequency"])
    sites = [str(s) for s in src["site_names"]]
    seg = {k: np.asarray(src[k][skill.start : skill.end]) for k in PER_FRAME}
    q_src = np.asarray(seg["qpos"], dtype=np.float64)
    n_lead = n_blend = 0
    if guard is not None and skill.kind == "guard":
        seg = derive_frames(np.stack([guard] * (len(q_src) + 2)), hz, sites)
    elif guard is not None:
        n_lead = min(skill.lead, len(q_src) // 2)
        if n_lead > 0:
            q_lead = lead_from_guard(q_src[:n_lead], guard)
            path = np.concatenate([q_lead[:1], q_lead, q_src[n_lead : n_lead + 1]])
            head = derive_frames(path, hz, sites)
            seg = {k: np.concatenate([head[k], seg[k][n_lead:]]) for k in PER_FRAME}
        n_blend = skill.blend
        if n_blend > 0:
            v_end = np.asarray(seg["qvel"][-1], dtype=np.float64)
            q_tail = blend_to_guard(q_src[-1], v_end, guard, n_blend, hz)
            path = np.concatenate([q_src[-1:], q_tail, q_tail[-1:]])
            tail = derive_frames(path, hz, sites)
            for k in VELOCITY_FIELDS:
                tail[k][-1] = 0.0  # arrives at rest
            seg = {k: np.concatenate([seg[k], tail[k]]) for k in PER_FRAME}
    for k in PER_FRAME:
        hold = np.repeat(seg[k][-1:], skill.pad, axis=0)
        if k in VELOCITY_FIELDS:
            hold = np.zeros_like(hold)
        out[k] = np.concatenate([seg[k], hold], axis=0).astype(src[k].dtype)
    total = out["qpos"].shape[0]
    out["split_points"] = np.array([0, total], dtype=np.int32)
    out["source_clip"] = np.str_(skill.clip)
    out["source_frames"] = np.array([skill.start, skill.end], dtype=np.int32)
    out["skill_kind"] = np.str_(skill.kind)
    out["lead_frames"] = np.int32(n_lead)
    out["blend_frames"] = np.int32(n_blend)
    return out


def cmd_cut(args: argparse.Namespace) -> None:
    """Write one .npz per manifest skill and check it loads as a MotionClip."""
    from myosuite.core.trajectory_io import check_clip_rate, load_motion_clip

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skills = load_manifest(args.manifest)
    guard = guard_qpos(skills)
    for skill in skills:
        path = args.out_dir / f"{skill.name}.npz"
        np.savez(path, **cut_skill(skill, guard))
        clip = load_motion_clip(path, expected_nq=89, expected_nv=88)
        check_clip_rate(clip.frequency_hz, 0.01, skill.name)
        print(
            f"{skill.name:26s} {skill.kind:8s} {skill.clip}[{skill.start}:{skill.end}]"
            f" + {skill.blend} to guard + {skill.pad} hold -> {clip.qpos.shape[0]} frames"
        )
    print(f"\n{len(skills)} skills written to {args.out_dir}")


def cmd_render(args: argparse.Namespace) -> None:
    """Render each skill kinematically (side view) and an all-skills reel."""
    import mujoco

    from myosuite.utils.video_io import add_text_to_frame, write_video

    model = fullbody_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 150.0, -8.0, 2.6
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reel: list[np.ndarray] = []
    skills = load_manifest(args.manifest)
    if args.only:
        skills = [s for s in skills if s.name in args.only]
    for skill in skills:
        clip = np.load(SKILL_DIR / f"{skill.name}.npz")
        qpos = clip["qpos"]
        n_motion = skill.end - skill.start
        n_blend = int(clip["blend_frames"]) if "blend_frames" in clip.files else 0
        n_lead = int(clip["lead_frames"]) if "lead_frames" in clip.files else 0
        frames = []
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [q[0], q[1], 1.0]
            renderer.update_scene(data, cam, scene_option=opt)
            if i < n_lead:
                tag = f"frame {skill.start + i} (from guard)"
            elif i < n_motion:
                tag = f"frame {skill.start + i}"
            else:
                tag = "to guard" if i < n_motion + n_blend else "hold"
            text = f"{skill.name} ({skill.kind})\n{skill.clip} {tag}"
            frames.append(
                np.asarray(
                    add_text_to_frame(
                        renderer.render(), text, color=(255, 255, 255), fontsize=16
                    )
                )
            )
        write_video(
            args.out_dir / f"{skill.name}.mp4", frames, fps=int(100 * args.speed)
        )
        reel.extend(frames + [frames[-1]] * 20)
        print(f"rendered {skill.name}")
    write_video(args.out_dir / "all_skills.mp4", reel, fps=int(100 * args.speed))
    renderer.close()
    print(f"Videos in {args.out_dir}")


def chain_frames(
    skills: list[Skill],
    q_start: np.ndarray | None = None,
    loops: int = 0,
    pause: int = 0,
    in_place: bool = False,
    label: str = "",
) -> Iterator[np.ndarray]:
    """Yield the ``qpos`` frames of skills played back to back.

    Args:
        skills: Skills to play in order.
        q_start: Pose whose root position and heading the first skill starts from
            (``None`` = the clip's own mocap position).
        loops: Passes over ``skills`` (0 = forever).
        pause: Frames to hold between skills.
        in_place: Start every skill from ``q_start`` instead of where the previous
            one ended, so repeated skills do not travel.
        label: Prefix for the printed skill names.

    Yields:
        One ``qpos`` of shape ``(89,)`` per frame.
    """
    q_last = q_start
    done = 0
    while loops == 0 or done < loops:
        for skill in skills:
            qpos = np.load(SKILL_DIR / f"{skill.name}.npz")["qpos"]
            if q_last is not None:
                qpos = anchor_qpos(qpos, q_start if in_place else q_last)
            q_last = qpos[-1]
            print(f"{label}{skill.name} ({skill.kind}), {len(qpos)} frames")
            yield from qpos
            yield from [qpos[-1]] * pause
        done += 1


def cmd_view(args: argparse.Namespace) -> None:
    """Loop skills kinematically in the interactive MuJoCo viewer."""
    import time

    import mujoco
    import mujoco.viewer

    by_name = {s.name: s for s in load_manifest(args.manifest)}
    unknown = [n for n in [*args.only, *args.opponent] if n not in by_name]
    if unknown:
        raise SystemExit(f"Unknown skills: {unknown}")
    skills = [by_name[n] for n in args.only] or list(by_name.values())
    pause = int(args.pause * 100)
    if args.vs:
        from myosuite import make_env

        env = make_env("myoChallengeBoxingVs-v0")
        env.reset(seed=0)
        model, data = env.unwrapped.model, env.unwrapped.data
        adr = [int(model.joint(f"{a}_root").qposadr[0]) for a in ("a0", "a1")]
        starts = [np.array(data.qpos[a : a + 89]) for a in adr]
        chains = [
            chain_frames(skills, starts[0], args.loops, pause, args.in_place, "a0: "),
            chain_frames(
                [by_name[n] for n in args.opponent], starts[1], 0, pause, True, "a1: "
            ),
        ]
    else:
        model = fullbody_model()
        data = mujoco.MjData(model)
        adr = [0]
        start = np.load(SKILL_DIR / f"{skills[0].name}.npz")["qpos"][0]
        chains = [chain_frames(skills, start, args.loops, pause, args.in_place)]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for frames in zip(*chains):
            if not viewer.is_running():
                break
            for a, q in zip(adr, frames):
                data.qpos[a : a + 89] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01 / args.speed)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run a subcommand."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="Suggest punch segments.")
    d.add_argument("--out", type=Path, default=None, help="Draft manifest path.")
    d.set_defaults(func=cmd_detect)
    c = sub.add_parser("cut", help="Cut manifest skills into .npz clips.")
    c.add_argument("--manifest", type=Path, default=MANIFEST)
    c.add_argument("--out-dir", type=Path, default=SKILL_DIR)
    c.set_defaults(func=cmd_cut)
    r = sub.add_parser("render", help="Render review videos.")
    r.add_argument("--manifest", type=Path, default=MANIFEST)
    r.add_argument("--out-dir", type=Path, default=VIDEO_DIR)
    r.add_argument(
        "--speed", type=float, default=0.5, help="0.5 = half-speed playback."
    )
    r.add_argument("--only", nargs="*", default=None, help="Render just these skills.")
    r.set_defaults(func=cmd_render)
    v = sub.add_parser("view", help="Loop skills in the interactive MuJoCo viewer.")
    v.add_argument("only", nargs="*", help="Skill names (default: all).")
    v.add_argument("--manifest", type=Path, default=MANIFEST)
    v.add_argument("--speed", type=float, default=1.0, help="0.5 = half speed.")
    v.add_argument("--pause", type=float, default=0.0, help="Seconds between skills.")
    v.add_argument(
        "--in-place",
        action="store_true",
        help="Restart every skill from the same spot instead of travelling.",
    )
    v.add_argument(
        "--vs", action="store_true", help="Play on agent a0 in the BoxingVs scene."
    )
    v.add_argument(
        "--opponent",
        nargs="*",
        default=["guard"],
        help="Skills agent a1 loops in place with --vs (default: guard).",
    )
    v.add_argument("--loops", type=int, default=0, help="0 = until the window closes.")
    v.set_defaults(func=cmd_view)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
