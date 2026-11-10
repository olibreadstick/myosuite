# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Boxing skill clips (cut by ``scripts/boxing_skills.py``) and their placement.

A skill clip is a MyoFullBody ``qpos`` trajectory that starts and ends in the
guard pose. Placing a clip moves it rigidly in the horizontal plane: a yaw
rotation about its first root position, then a translation of that root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[6]
SKILL_DIR = _REPO_ROOT / "boxing_motions/MyoFullBody/gmr/skills"
MANIFEST = Path(__file__).with_name("boxing_skills.yaml")

NQ = 89  # MyoFullBody qpos width (free root + joints)
SITE_KEYS = ("fist_r", "fist_l", "head_zone", "body_zone", "pelvis_site")
KINDS = ("guard", "punch_R", "punch_L", "combo", "step_fwd", "step_back", "special")


@dataclass(frozen=True)
class SkillClip:
    """A skill clip in its own (mocap) coordinates, with precomputed site paths.

    Attributes:
        name: Skill name from the manifest.
        kind: Menu group (one of :data:`KINDS`).
        qpos: ``(n, 89)`` poses.
        sites: ``(n, len(SITE_KEYS), 3)`` world positions of the boxing sites.
        fist_speed: ``(n, 2)`` right/left fist speed (m/s).
        forward: ``(2,)`` horizontal pelvis forward direction at frame 0.
        hz: Frame rate.
    """

    name: str
    kind: str
    qpos: np.ndarray
    sites: np.ndarray
    fist_speed: np.ndarray
    forward: np.ndarray
    hz: float

    @property
    def n(self) -> int:
        """Number of frames."""
        return int(self.qpos.shape[0])


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    """Read the skill manifest: ``{name: {clip, frames, kind, ...}}``."""
    return yaml.safe_load(path.read_text())["skills"]


def yaw_of(quat: np.ndarray) -> float:
    """Heading angle (rad, about world z) of a wxyz quaternion."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Heading-only part (rotation about world z) of a wxyz quaternion."""
    yaw = np.array([quat[0], 0.0, 0.0, quat[3]])
    return yaw / np.linalg.norm(yaw)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def place_points(
    points: np.ndarray, origin: np.ndarray, xy: np.ndarray, dyaw: float
) -> np.ndarray:
    """Rotate points by ``dyaw`` about ``origin`` and move ``origin`` to ``xy``.

    Args:
        points: ``(..., 3)`` positions.
        origin: ``(3,)`` pivot (the clip's first root position).
        xy: ``(2,)`` new horizontal position of the pivot.
        dyaw: Rotation about world z (rad).

    Returns:
        Moved positions (height unchanged).
    """
    out = (np.asarray(points, np.float64) - origin) @ _rot_z(dyaw).T
    out[..., :2] += xy
    out[..., 2] += origin[2]
    return out


def place_qpos(qpos: np.ndarray, xy: np.ndarray, dyaw: float) -> np.ndarray:
    """Move a clip rigidly: rotate by ``dyaw`` about its first root, then translate it.

    Args:
        qpos: ``(n, 89)`` poses.
        xy: ``(2,)`` horizontal position of the first root.
        dyaw: Rotation about world z (rad).

    Returns:
        Moved copy of ``qpos``.
    """
    out = np.asarray(qpos, np.float64).copy()
    out[:, :3] = place_points(out[:, :3], out[0, :3].copy(), xy, dyaw)
    # Pre-multiply every root quaternion by the yaw rotation.
    c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)
    w, x, y, z = out[:, 3].copy(), out[:, 4].copy(), out[:, 5].copy(), out[:, 6].copy()
    out[:, 3] = c * w - s * z
    out[:, 4] = c * x - s * y
    out[:, 5] = c * y + s * x
    out[:, 6] = c * z + s * w
    return out


def anchor_qpos(qpos: np.ndarray, q_from: np.ndarray) -> np.ndarray:
    """Move a clip so its first frame sits at the root position and heading of ``q_from``.

    Args:
        qpos: Clip poses ``(n, nq)``.
        q_from: Pose to continue from (e.g. the previous skill's last frame).

    Returns:
        Re-anchored copy of ``qpos``.
    """
    dyaw = yaw_of(q_from[3:7]) - yaw_of(qpos[0, 3:7])
    return place_qpos(qpos, np.asarray(q_from[:2], np.float64), dyaw)


def load_skill_clips(
    model: mujoco.MjModel,
    site_ids: dict[str, int],
    qpos_adr: int,
    pelvis_body: int,
    names: list[str] | None = None,
    skill_dir: Path = SKILL_DIR,
    manifest: Path = MANIFEST,
) -> dict[str, SkillClip]:
    """Load skill clips and precompute their boxing-site paths by forward kinematics.

    Args:
        model: Model containing one MyoFullBody boxer (e.g. the BoxingVs model).
        site_ids: That boxer's site ids by :data:`SITE_KEYS` key.
        qpos_adr: Address of that boxer's 89 ``qpos`` values.
        pelvis_body: That boxer's pelvis body id (its x axis is the forward direction).
        names: Skills to load (default: every manifest skill).
        skill_dir: Folder of the cut ``.npz`` clips.
        manifest: Skill manifest.

    Returns:
        Clips by name.
    """
    entries = load_manifest(manifest)
    data = mujoco.MjData(model)
    sid = [site_ids[k] for k in SITE_KEYS]
    clips = {}
    for name in names or list(entries):
        npz = np.load(skill_dir / f"{name}.npz", allow_pickle=True)
        qpos = np.asarray(npz["qpos"], dtype=np.float64)
        hz = float(npz["frequency"])
        sites = np.empty((len(qpos), len(sid), 3))
        forward = np.zeros(2)
        for i, q in enumerate(qpos):
            data.qpos[qpos_adr : qpos_adr + NQ] = q
            mujoco.mj_forward(model, data)
            sites[i] = data.site_xpos[sid]
            if i == 0:
                forward = data.xmat[pelvis_body].reshape(3, 3)[:2, 0].copy()
        speed = np.linalg.norm(np.diff(sites[:, :2], axis=0), axis=2) * hz
        clips[name] = SkillClip(
            name=name,
            kind=str(entries[name]["kind"]),
            qpos=qpos,
            sites=sites,
            fist_speed=np.concatenate([speed[:1], speed]),
            forward=forward / np.linalg.norm(forward),
            hz=hz,
        )
    return clips
