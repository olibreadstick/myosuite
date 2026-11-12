"""Checks for the boxing skill library manifest and cutter (scripts/boxing_skills.py)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "boxing_skills.py"

spec = importlib.util.spec_from_file_location("boxing_skills", SCRIPT)
boxing_skills = importlib.util.module_from_spec(spec)
sys.modules["boxing_skills"] = boxing_skills  # dataclasses look the module up here
spec.loader.exec_module(boxing_skills)

pytestmark = pytest.mark.skipif(
    not boxing_skills.CLIP_DIR.is_dir(), reason="boxing mocap clips not checked out"
)


def test_manifest_ranges_fit_their_clips() -> None:
    skills = boxing_skills.load_manifest()
    assert len({s.name for s in skills}) == len(skills)
    for s in skills:
        n = np.load(boxing_skills.CLIP_DIR / f"{s.clip}.npz")["qpos"].shape[0]
        assert 0 <= s.start < s.end <= n, s.name
        assert s.kind, s.name


def test_cut_slices_and_pads() -> None:
    skill = next(s for s in boxing_skills.load_manifest() if s.kind == "punch_R")
    src = np.load(boxing_skills.CLIP_DIR / f"{skill.clip}.npz")
    out = boxing_skills.cut_skill(skill)
    n = skill.end - skill.start
    for k in boxing_skills.PER_FRAME:
        assert out[k].shape[0] == n + skill.pad, k
        np.testing.assert_array_equal(out[k][:n], src[k][skill.start : skill.end])
    np.testing.assert_array_equal(
        out["qpos"][n:], np.repeat(out["qpos"][n - 1 : n], skill.pad, 0)
    )
    assert not out["qvel"][n:].any() and not out["cvel"][n:].any()
    assert list(out["split_points"]) == [0, n + skill.pad]


def test_cut_clip_loads(tmp_path: Path) -> None:
    from myosuite.core.trajectory_io import load_motion_clip

    skill = boxing_skills.load_manifest()[0]
    path = tmp_path / f"{skill.name}.npz"
    np.savez(path, **boxing_skills.cut_skill(skill))
    clip = load_motion_clip(path, expected_nq=89, expected_nv=88)
    assert clip.frequency_hz == 100.0
    assert clip.qpos.shape[0] == skill.end - skill.start + skill.pad


def test_detect_finds_alternating_punches() -> None:
    d = np.load(boxing_skills.CLIP_DIR / "punchboxing_push_poses.npz")
    punches = boxing_skills.detect_punches(
        d, boxing_skills.PUNCH_WINDOWS["punchboxing_push_poses"]
    )
    assert [p["hand"] for p in punches] == ["R", "L", "R"]
    for a, b in itertools.pairwise(punches):
        assert a["end"] <= b["start"]


def test_cut_starts_and_ends_in_guard() -> None:
    skills = boxing_skills.load_manifest()
    guard = boxing_skills.guard_qpos(skills)
    skill = next(s for s in skills if s.name == "punch_R_push1")
    src = np.load(boxing_skills.CLIP_DIR / f"{skill.clip}.npz")
    out = boxing_skills.cut_skill(skill, guard)
    n, lead = skill.end - skill.start, int(out["lead_frames"])
    assert lead == skill.lead
    for k in boxing_skills.PER_FRAME:
        assert out[k].shape[0] == n + skill.blend + skill.pad, k
        np.testing.assert_array_equal(
            out[k][lead:n], src[k][skill.start + lead : skill.end]
        )
    q = out["qpos"].astype(np.float64)
    step = np.abs(np.diff(q[:, 7:], axis=0)).max(axis=1)
    # Starts and ends in the guard's joint pose and height, at rest at the end.
    for i in (0, -1):
        np.testing.assert_allclose(q[i, 7:], guard[7:], atol=1e-6)
        np.testing.assert_allclose(q[i, 2], guard[2], atol=1e-6)
    assert not out["qvel"][n + skill.blend - 1 :].any()
    # No jump where generated frames meet the mocap.
    assert step[lead - 1] <= 1.5 * step[lead]
    assert step[n - 1] <= 1.5 * step[n - 2]
    # The root stays where the punch ended instead of sliding to the guard's spot.
    assert np.linalg.norm(q[-1, :2] - q[n - 1, :2]) < 0.1


def test_anchor_continues_from_previous_skill() -> None:
    skills = boxing_skills.load_manifest()
    guard = boxing_skills.guard_qpos(skills)
    a, b = (
        boxing_skills.cut_skill(s, guard)["qpos"].astype(np.float64)
        for s in skills
        if s.name in ("punch_R_push1", "punch_L_stand1")
    )
    moved = boxing_skills.anchor_qpos(b, a[-1])
    np.testing.assert_allclose(moved[0], a[-1], atol=1e-5)
    np.testing.assert_allclose(moved[:, 7:], b[:, 7:])
    np.testing.assert_allclose(moved[:, 2], b[:, 2])


def test_guard_joint_overrides() -> None:
    skills = boxing_skills.load_manifest()
    guard = next(s for s in skills if s.name == "guard")
    base = boxing_skills.guard_qpos(skills)
    guard.joints = {"elbow_flex_r": 2.0}
    adr = boxing_skills.fullbody_model().joint("elbow_flex_r").qposadr[0]
    changed = boxing_skills.guard_qpos(skills)
    assert changed[adr] == 2.0
    np.testing.assert_array_equal(np.delete(changed, adr), np.delete(base, adr))
    assert boxing_skills.cut_skill(guard, changed)["qpos"][0, adr] == np.float32(2.0)
