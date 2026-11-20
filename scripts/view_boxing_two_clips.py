#!/usr/bin/env python3
"""Visualize lunge_push.npz (red) against a standing NPZ (blue).

Both agents are KINEMATICALLY driven by their clips: their qpos is imposed every
frame, so inter-agent contact can be detected but neither can be pushed over.
This is for inspecting relative placement and the timing of contact, not
for testing physical recovery or a trained policy.

From myosuite-capstone:
    uv run mjpython scripts/view_boxing_two_clips.py --separation 1.0 --seconds 6
"""

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from myosuite.core.trajectory_io import load_motion_clip
from myosuite.envs.myo.tasks.challenge.boxing_vs import BoxingVsConfig, BoxingVsEnv
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "boxing_motions" / "MyoFullBody" / "gmr"
RED_DEFAULT = CLIPS / "skills" / "lunge_push.npz"
BLUE_DEFAULT = CLIPS / "Transitions_mocap" / "mazen_c3d" / "punchboxing_stand_poses.npz"


def rotate(quaternion, vec):
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, np.asarray(vec, dtype=np.float64),
                         np.asarray(quaternion, dtype=np.float64))
    return result


def multiply_quaternions(a, b):
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=np.float64),
                      np.asarray(b, dtype=np.float64))
    return out


def agent_joint_indices(model, joint_ids):
    # MuJoCo joint enum: free=0, ball=1, slide=2, hinge=3.
    qwidth = (7, 4, 1, 1)
    vwidth = (6, 3, 1, 1)
    qs, vs = [], []
    for joint in joint_ids:
        kind = int(model.jnt_type[joint])
        qadr = int(model.jnt_qposadr[joint])
        vadr = int(model.jnt_dofadr[joint])
        qs.extend(range(qadr, qadr + qwidth[kind]))
        vs.extend(range(vadr, vadr + vwidth[kind]))
    return np.asarray(qs, dtype=int), np.asarray(vs, dtype=int)


def place_clip_frame(clip_q, clip_v, root_q, root_v, yaw, anchor, source_root):
    """Rotate and translate the clip's initial root into the selected arena anchor."""
    qyaw = np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    q = np.asarray(clip_q, dtype=np.float64).copy()
    v = np.asarray(clip_v, dtype=np.float64).copy()
    q[root_q:root_q+3] = anchor + rotate(qyaw, q[root_q:root_q+3] - source_root)
    q[root_q+3:root_q+7] = multiply_quaternions(qyaw, clip_q[root_q+3:root_q+7])
    v[root_v:root_v+3] = rotate(qyaw, v[root_v:root_v+3])
    v[root_v+3:root_v+6] = rotate(qyaw, v[root_v+3:root_v+6])
    return q, v


def get_frame(clip, seconds_since_start, freeze_first=False):
    if freeze_first:
        return 0
    return min(int(round(max(0.0, seconds_since_start) * float(clip.frequency_hz))),
               len(clip.qpos) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--red-clip", type=Path, default=RED_DEFAULT)
    parser.add_argument("--blue-clip", type=Path, default=BLUE_DEFAULT)
    parser.add_argument("--separation", type=float, default=1.0,
                        help="Initial root-to-root distance in metres")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--warmup", type=float, default=0.4,
                        help="Time before the red lunge begins")
    parser.add_argument("--freeze-blue", action="store_true",
                        help="Hold the first frame of the blue standing clip")
    args = parser.parse_args()
    if args.separation <= 0 or args.seconds <= 0 or args.warmup < 0:
        parser.error("Separation and seconds must be positive; warmup cannot be negative")
    for path in (args.red_clip, args.blue_clip):
        if not path.is_file():
            parser.error(f"Missing NPZ clip: {path}")

    config = BoxingVsConfig(
        cross_agent_collisions=True,
        massless_visual_gear=True,
        disable_fingers=True,
        agent_separation_m=args.separation,
    )
    env = BoxingVsEnv(config=config)
    try:
        env.reset(seed=0)
        model, data, meta = env.model, env.data, env._meta
        single_cfg = default_mimic_fullbody_config()
        single_cfg.disable_fingers = True
        fb_model, _, _ = compile_mimic_fullbody_mjmodel(single_cfg)
        root_id = mujoco.mj_name2id(fb_model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_id < 0:
            raise RuntimeError("Standalone model has no root joint")
        root_q = int(fb_model.jnt_qposadr[root_id])
        root_v = int(fb_model.jnt_dofadr[root_id])

        red = load_motion_clip(args.red_clip, fb_model.nq, fb_model.nv)
        blue = load_motion_clip(args.blue_clip, fb_model.nq, fb_model.nv)
        if red.qvel is None or blue.qvel is None:
            raise ValueError("Both NPZ clips must contain qvel")
        qa, va = agent_joint_indices(model, meta.jnt_ids["agent_0"])
        qb, vb = agent_joint_indices(model, meta.jnt_ids["agent_1"])
        if (len(qa), len(va), len(qb), len(vb)) != (fb_model.nq, fb_model.nv, fb_model.nq, fb_model.nv):
            raise ValueError("Agent joint sizes differ from standalone model; check model settings")

        # Aim the primary displacement of the lunge from +Y toward -Y.
        red_start = np.asarray(red.qpos[0, root_q:root_q+3], dtype=float)
        displacement = np.asarray(red.qpos[:, root_q:root_q+2], dtype=float) - red_start[:2]
        primary = displacement[np.argmax(np.linalg.norm(displacement, axis=1))]
        red_yaw = (-math.pi / 2 - math.atan2(primary[1], primary[0])
                   if np.linalg.norm(primary) > 0.05 else 0.0)
        blue_yaw = math.pi  # Face red agent.
        blue_start = np.asarray(blue.qpos[0, root_q:root_q+3], dtype=float)
        red_anchor = np.array([0.0, args.separation / 2, red_start[2]])
        blue_anchor = np.array([0.0, -args.separation / 2, blue_start[2]])

        dt = float(config.ctrl_dt)
        frames = int(math.ceil(args.seconds / dt))
        contact_steps = 0
        max_contacts = 0
        print(f"Red: {args.red_clip.name} ({len(red.qpos)} frames, {red.frequency_hz} Hz)")
        print(f"Blue: {args.blue_clip.name} ({len(blue.qpos)} frames, {blue.frequency_hz} Hz)")
        print(f"Initial separation: {args.separation:.2f} m; red lunge begins at {args.warmup:.2f} s")
        print("Both agents follow their NPZ poses directly (no learned control, no falling).")
        print("Contact detection enabled; kinematic contact does not produce balance reactions.")
        print("Close viewer or press ESC to exit.")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            for k in range(frames):
                if not viewer.is_running():
                    break
                tick = time.perf_counter()
                seconds = k * dt
                rframe = get_frame(red, seconds - args.warmup,
                                   freeze_first=seconds < args.warmup)
                bframe = get_frame(blue, seconds, freeze_first=args.freeze_blue)
                rq, rv = place_clip_frame(red.qpos[rframe], red.qvel[rframe],
                                          root_q, root_v, red_yaw, red_anchor, red_start)
                bq, bv = place_clip_frame(blue.qpos[bframe], blue.qvel[bframe],
                                          root_q, root_v, blue_yaw, blue_anchor, blue_start)
                if seconds < args.warmup or rframe == len(red.qpos) - 1:
                    rv[:] = 0.0
                if args.freeze_blue or bframe == len(blue.qpos) - 1:
                    bv[:] = 0.0
                data.qpos[qa], data.qvel[va] = rq, rv
                data.qpos[qb], data.qvel[vb] = bq, bv
                data.ctrl[:] = 0.0
                data.time = seconds
                # Forward kinematics + contact detection only; no physics integration.
                mujoco.mj_forward(model, data)

                contacts = 0
                for i in range(data.ncon):
                    c = data.contact[i]
                    if {int(meta.geom_agent[c.geom1]), int(meta.geom_agent[c.geom2])} == {0, 1}:
                        contacts += 1
                if contacts:
                    contact_steps += 1
                    max_contacts = max(max_contacts, contacts)
                if k % 50 == 0 or (contacts and contact_steps == 1):
                    print(f"t={seconds:5.2f}s  red_frame={rframe:3}  "
                          f"blue_frame={bframe:3}  agent_contacts={contacts}")
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - tick)))

        print(f"Done: contact at {contact_steps} frames; peak simultaneous contacts: {max_contacts}")
        if not contact_steps:
            print("No inter-agent contact detected. Try --separation 0.8 or 0.6.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
