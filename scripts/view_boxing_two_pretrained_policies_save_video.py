#!/usr/bin/env python3
"""Two *dynamic* MuscleMimic policies in the same MyoSuite BoxingVs world.

Red: mm-motion-100 policy tracking the lunge_push.npz motion reference.
Blue: independent mm-motion-100 policy tracking a held standing reference.

Both agents are simulated by MuJoCo.  No qpos/qvel are imposed after initial
placement, and contacts can displace both. A pretrained generalist tracker is
NOT automatically a trained push/balance-recovery skill; validate behaviour.

Requires the already working scripts/view_boxing_lunge_push.py, which defines
FullbodyTrackingObsAdapter (2,400-d observation goal for mm-motion-100).

Run from myosuite-capstone:
    uv run --with orbax-checkpoint --with 'mujoco-mjx==3.8.1' --with flax \
      mjpython scripts/view_boxing_two_pretrained_policies.py \
      --checkpoint ../musclemimic_mcgill_dev/mm-motion-100
"""

import argparse
import math
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from myosuite.core.trajectory_io import load_motion_clip, expand_motion_clip_to_model
from myosuite.envs.myo.tasks.challenge.boxing_vs import BoxingVsConfig, BoxingVsEnv
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    LocalPolicyRunner,
    fullbody_history_settings_from_metadata,
    load_local_policy_artifacts,
    read_checkpoint_config_metadata,
)
from view_boxing_lunge_push import (
    FullbodyTrackingObsAdapter,
    agent_indices,
    place_clip_frame,
    root_addresses,
    rotate,
    quat_mul,
)

HERE = Path(__file__).resolve().parents[1]
MOTIONS = HERE / "boxing_motions" / "MyoFullBody" / "gmr"
RED_DEFAULT = MOTIONS / "skills" / "lunge_push.npz"
BLUE_DEFAULT = MOTIONS / "skills" / "sit_stand_last_1s.npz"
BLUE_FALLBACK = MOTIONS / "skills" / "standing_idle.npz"
BLUE_ORIGINAL = MOTIONS / "Transitions_mocap" / "mazen_c3d" / "sit_stand_poses.npz"
CHECKPOINT_DEFAULT = HERE.parent / "musclemimic_mcgill_dev" / "mm-motion-100"


def to_local(world_q, world_v, root_q, root_v, yaw, anchor, reference_root):
    """Transform a world's root into the reference clip coordinate system.

    MuJoCo free-joint linear velocity is in world coordinates; free-joint
    angular velocity is in the body-local frame, so it must NOT be yaw-rotated.
    """
    inv_yaw = np.array([math.cos(yaw / 2), 0, 0, -math.sin(yaw / 2)])
    q = np.asarray(world_q, dtype=np.float64).copy()
    v = np.asarray(world_v, dtype=np.float64).copy()
    q[root_q : root_q + 3] = reference_root + rotate(
        inv_yaw, q[root_q : root_q + 3] - anchor
    )
    q[root_q + 3 : root_q + 7] = quat_mul(
        inv_yaw, world_q[root_q + 3 : root_q + 7]
    )
    v[root_v : root_v + 3] = rotate(inv_yaw, world_v[root_v : root_v + 3])
    # v[root_v + 3:root_v + 6] stays body-local, unchanged by world yaw.
    return q, v


def prepare_reference(
    source_path, model, site_names, dt, duration, pad, mode, delay, destination
):
    """Build complete, time-aligned NPZ for both policy and tracking adapter.

    This regenerates only simulation-derived kinematics in a trusted temporary
    NPZ. It also works with user-extracted clips containing only qpos/qvel.
    """
    source = expand_motion_clip_to_model(
        load_motion_clip(source_path, model.nq, model.nv), model
    )
    if source.qpos is None or source.qvel is None:
        raise ValueError(f"Motion clip must include both qpos and qvel: {source_path}")
    fps = float(source.frequency_hz or 0)
    expected = 1.0 / dt
    if not np.isclose(fps, expected, rtol=0.015):
        raise ValueError(
            f"{source_path.name} is {fps:g} Hz but this checkpoint uses "
            f"{expected:g} Hz. Resample the clip before this experiment."
        )
    steps = int(math.ceil(duration / dt)) + pad
    qpos = np.empty((steps, model.nq), dtype=np.float64)
    qvel = np.empty((steps, model.nv), dtype=np.float64)
    motion = np.asarray(source.qpos, dtype=np.float64)
    velocity = np.asarray(source.qvel, dtype=np.float64)
    if mode == "hold-last":
        qpos[:] = motion[-1]
        qvel[:] = 0
    elif mode == "follow":
        frames = np.minimum(np.arange(steps), len(motion) - 1)
        qpos[:] = motion[frames]
        qvel[:] = velocity[frames]
        qvel[frames == len(motion) - 1] = 0  # hold the last pose, not its velocity
    elif mode == "delay-follow":
        warmup = int(round(delay / dt))
        if warmup >= steps:
            raise ValueError("Delay is longer than the simulation")
        frames = np.clip(np.arange(steps) - warmup, 0, len(motion) - 1)
        qpos[:] = motion[frames]
        qvel[:] = velocity[frames]
        qvel[:warmup] = 0
        qvel[np.arange(steps) - warmup >= len(motion) - 1] = 0
    else:
        raise ValueError(f"Unknown reference mode: {mode}")

    # MuscleMimic fullbody tracking is root-XY-relative to the start of the
    # reference. Build the temporary reference at XY=(0,0) so the current
    # stand-alone shadow state and reference site positions use the SAME frame.
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_joint < 0:
        raise ValueError("Missing free-joint root")
    root_adr = int(model.jnt_qposadr[root_joint])
    qpos[:, root_adr:root_adr+2] -= qpos[0, root_adr:root_adr+2].copy()

    # Legacy adapter used for the joint/muscle state block expects site
    # kinematics in the NPZ. Recompute them for the current single-agent model.
    # The *new* tracking goal also recomputes its reference kinematics via mj_forward.
    site_ids = []
    for name in site_names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise ValueError(f"Tracking site {name!r} missing from model")
        site_ids.append(idx)
    site_ids = np.asarray(site_ids, dtype=np.int32)
    nsites = len(site_ids)
    site_xpos = np.empty((steps, nsites, 3), dtype=np.float32)
    site_xmat = np.empty((steps, nsites, 9), dtype=np.float32)
    cvel = np.empty((steps, model.nbody, 6), dtype=np.float32)
    subtree_com = np.empty((steps, model.nbody, 3), dtype=np.float32)
    d = mujoco.MjData(model)
    # For repeated held poses, avoid recomputing expensive fullbody kinematics.
    states_cache = {}
    for i in range(steps):
        key = (qpos[i].tobytes(), qvel[i].tobytes())
        cached = states_cache.get(key)
        if cached is None:
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            mujoco.mj_forward(model, d)
            cached = (
                np.array(d.site_xpos[site_ids], dtype=np.float32),
                np.array(d.site_xmat[site_ids], dtype=np.float32),
                np.array(d.cvel, dtype=np.float32),
                np.array(d.subtree_com, dtype=np.float32),
            )
            states_cache[key] = cached
        site_xpos[i], site_xmat[i], cvel[i], subtree_com[i] = cached
    np.savez_compressed(
        destination,
        qpos=qpos,
        qvel=qvel,
        frequency=np.array(fps, dtype=np.float64),
        site_xpos=site_xpos,
        site_xmat=site_xmat,
        cvel=cvel,
        subtree_com=subtree_com,
        site_bodyid=np.asarray(model.site_bodyid[site_ids], dtype=np.int32),
        body_rootid=np.asarray(model.body_rootid, dtype=np.int32),
        site_names=np.asarray(site_names, dtype=str),
    )
    return load_motion_clip(destination, model.nq, model.nv)


def make_controller(model, clip, checkpoint, seed, role):
    meta = read_checkpoint_config_metadata(checkpoint)
    if meta.get("experiment", {}).get("env_params", {}).get("goal_type") not in (
        "GoalFullBodyTracking", "GoalFullBodyTrackingVisual"
    ):
        raise ValueError(
            f"{role}: This script expects a GoalFullBodyTracking checkpoint."
        )
    artifacts = load_local_policy_artifacts(checkpoint)
    if artifacts.action_dim != model.nu:
        raise ValueError(
            f"{role}: action dim {artifacts.action_dim} != model muscles {model.nu}"
        )
    adapter = FullbodyTrackingObsAdapter(model, clip, meta)
    check = mujoco.MjData(model)
    check.qpos[:] = clip.qpos[0]
    check.qvel[:] = clip.qvel[0]
    mujoco.mj_forward(model, check)
    observation = adapter.build(check, 0)
    print(
        f"{role}: observation={len(observation)} "
        f"checkpoint={artifacts.obs_dim} goal={adapter.goal_dim} "
        f"actions={artifacts.action_dim}"
    )
    if len(observation) != artifacts.obs_dim:
        raise ValueError(f"{role}: Observation size mismatch; will not feed policy")
    runner = LocalPolicyRunner(
        artifacts=artifacts,
        stochastic=False,
        seed=seed,
        obs_adapter=adapter,
        **fullbody_history_settings_from_metadata(meta),
    )
    return runner, mujoco.MjData(model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT,
                   help="Checkpoint shared by both policies, default mm-motion-100")
    p.add_argument("--red-checkpoint", type=Path, default=None,
                   help="Optional separate pretrained checkpoint for red")
    p.add_argument("--blue-checkpoint", type=Path, default=None,
                   help="Optional separate pretrained checkpoint for blue")
    p.add_argument("--red-clip", type=Path, default=RED_DEFAULT)
    p.add_argument("--blue-clip", type=Path, default=BLUE_DEFAULT)
    p.add_argument("--blue-follow-clip", action="store_true",
                   help="Follow its actual sequence instead of holding the final standing frame")
    p.add_argument("--separation", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--red-delay", type=float, default=1.0,
                   help="Delay red pushing goal for a blue-only standing baseline")
    p.add_argument("--no-viewer", action="store_true", help="Run headless for debugging")
    p.add_argument("--replay", action="store_true",
                   help="Simulate both dynamic policies first, then replay their actual physics states at normal speed")
    p.add_argument("--replay-fps", type=int, default=50,
                   help="Target viewer replay frame rate (default 50)")
    p.add_argument("--save-rollout", type=Path, default=None,
                   help="Optional output NPZ of simulated agent states for analysis")
    p.add_argument("--save-video", type=Path, default=None,
                   help="Export simulated rollout as MP4 after physics simulation")
    p.add_argument("--video-fps", type=int, default=30,
                   help="Exported MP4 frame rate (default: 30)")
    args = p.parse_args()
    if args.blue_clip == BLUE_DEFAULT and not args.blue_clip.is_file():
        if BLUE_FALLBACK.is_file():
            args.blue_clip = BLUE_FALLBACK
        elif BLUE_ORIGINAL.is_file():
            args.blue_clip = BLUE_ORIGINAL
    if args.replay_fps < 1 or args.video_fps < 1:
        p.error("--replay-fps and --video-fps must be at least 1")
    if args.seconds <= 0 or args.separation <= 0 or not 0 <= args.red_delay < args.seconds:
        p.error("Require seconds/separation > 0 and 0 <= red-delay < seconds")
    for f in (args.blue_clip, args.red_clip, args.red_checkpoint or args.checkpoint,
              args.blue_checkpoint or args.checkpoint):
        if not Path(f).exists():
            p.error(f"File or checkpoint not found: {f}")

    cfg = BoxingVsConfig(
        cross_agent_collisions=True,
        massless_visual_gear=True,
        disable_fingers=True,
        agent_separation_m=args.separation,
    )
    env = BoxingVsEnv(config=cfg)
    try:
        env.reset(seed=0)
        model, data, meta = env.model, env.data, env._meta
        single_cfg = default_mimic_fullbody_config()
        single_cfg.disable_fingers = True
        fb_model, _, _ = compile_mimic_fullbody_mjmodel(single_cfg)
        rq, rv = root_addresses(fb_model)
        dt = float(cfg.ctrl_dt)
        steps = int(math.ceil(args.seconds / dt))
        frames_pad = 100  # >= max lookahead (80) in mm-motion-100

        if args.red_checkpoint or args.blue_checkpoint:
            # The reference site list may vary across separately trained policies.
            m_red = read_checkpoint_config_metadata(args.red_checkpoint or args.checkpoint)
            m_blue = read_checkpoint_config_metadata(args.blue_checkpoint or args.checkpoint)
        else:
            m_red = m_blue = read_checkpoint_config_metadata(args.checkpoint)
        def sites(m):
            return m["experiment"]["env_params"]["goal_params"]["sites_for_mimic"]

        with tempfile.TemporaryDirectory(prefix="mm_two_policies_") as folder:
            folder = Path(folder)
            print("Preparing red dynamic tracking reference...")
            red = prepare_reference(
                args.red_clip, fb_model, sites(m_red), dt, args.seconds, frames_pad,
                "delay-follow", args.red_delay, folder / "red_reference.npz"
            )
            print("Preparing blue standing tracking reference...")
            blue = prepare_reference(
                args.blue_clip, fb_model, sites(m_blue), dt, args.seconds, frames_pad,
                "follow" if args.blue_follow_clip else "hold-last", 0,
                folder / "blue_reference.npz"
            )

            print("Loading red pretrained policy...")
            red_policy, red_shadow = make_controller(
                fb_model, red, args.red_checkpoint or args.checkpoint, 1, "RED"
            )
            print("Loading blue pretrained policy...")
            blue_policy, blue_shadow = make_controller(
                fb_model, blue, args.blue_checkpoint or args.checkpoint, 2, "BLUE"
            )
            q_idx = []
            v_idx = []
            act_idx = []
            for agent_no in (0, 1):
                qa, va = agent_indices(model, meta.jnt_ids[f"agent_{agent_no}"])
                ai = np.asarray(meta.act_indices[f"agent_{agent_no}"], dtype=int)
                if len(qa) != fb_model.nq or len(va) != fb_model.nv or len(ai) != fb_model.nu:
                    raise ValueError("Combined per-agent indices do not match shadow model")
                q_idx.append(qa)
                v_idx.append(va)
                act_idx.append(ai)

            # Preserve the proven placement convention of view_boxing_lunge_push.
            original_red = expand_motion_clip_to_model(
                load_motion_clip(args.red_clip, fb_model.nq, fb_model.nv), fb_model
            )
            red_start = np.asarray(red.qpos[0, rq:rq+3], dtype=float)
            original_start = np.asarray(original_red.qpos[0, rq:rq+3], dtype=float)
            displacement = original_red.qpos[:, rq:rq+2] - original_start[:2]
            biggest = displacement[np.argmax(np.linalg.norm(displacement, axis=1))]
            red_yaw = (
                -math.pi / 2 - math.atan2(biggest[1], biggest[0])
                if np.linalg.norm(biggest) > 0.05 else 0.0
            )
            blue_start = np.asarray(blue.qpos[0, rq:rq+3], dtype=float)
            yaws = (red_yaw, math.pi)
            starts = (red_start, blue_start)
            anchors = (
                np.array([0.0, args.separation / 2, red_start[2]]),
                np.array([0.0, -args.separation / 2, blue_start[2]]),
            )
            clips = (red, blue)
            controllers = (red_policy, blue_policy)
            shadows = (red_shadow, blue_shadow)

            # Initialize ONCE. Never force qpos/qvel again during the rollout.
            mujoco.mj_resetData(model, data)
            for i in (0, 1):
                q, v = place_clip_frame(
                    clips[i].qpos[0], clips[i].qvel[0], rq, rv,
                    yaws[i], anchors[i], starts[i]
                )
                data.qpos[q_idx[i]] = q
                data.qvel[v_idx[i]] = v
            mujoco.mj_forward(model, data)
            print(f"Red: pretrained policy tracking {args.red_clip.name}")
            print(f"Blue: pretrained policy tracking "
                  f"{'full clip' if args.blue_follow_clip else 'last standing pose of'} "
                  f"{args.blue_clip.name}")
            print(f"Red goal delay: {args.red_delay:g}s; separation: {args.separation:g}m")
            print("Both agents: free dynamics + muscle actions, collision enabled; NO pose forcing")
            print("Note: observations use exploratory bridge; validate parity for quantitative studies")
            last_report = -1
            contacts_steps = 0
            initial_blue_xy = np.asarray(data.qpos[q_idx[1]][rq:rq+2]).copy()
            initial_red_xy = np.asarray(data.qpos[q_idx[0]][rq:rq+2]).copy()
            max_blue_displacement = 0.0
            max_red_displacement = 0.0
            peak_pair_contact_force = 0.0
            max_contacts = 0
            blue_lowest_pelvis = float(data.xpos[meta.pelvis_body_id["agent_1"], 2])
            # Record the actual dynamic rollout. A replay only restores observed states
            # for display; it does not replace either policy during simulation.
            record_states = (args.replay or args.save_rollout is not None
                             or args.save_video is not None)
            recorded = []
            if record_states:
                recorded.append((data.qpos.copy(), data.qvel.copy(),
                                 data.act.copy(), data.ctrl.copy()))

            def simulate(viewer=None):
                nonlocal last_report, contacts_steps, max_blue_displacement
                nonlocal max_red_displacement, peak_pair_contact_force, max_contacts
                nonlocal blue_lowest_pelvis
                wall_start = time.perf_counter()
                completed_steps = 0
                for k in range(steps):
                    if viewer is not None and not viewer.is_running():
                        break
                    tick = time.perf_counter()
                    for i in (0, 1):
                        q, v = to_local(
                            data.qpos[q_idx[i]], data.qvel[v_idx[i]], rq, rv,
                            yaws[i], anchors[i], starts[i]
                        )
                        shadow = shadows[i]
                        shadow.qpos[:] = q
                        shadow.qvel[:] = v
                        if fb_model.na:
                            if fb_model.na != len(act_idx[i]):
                                raise ValueError("Cannot map muscle activation states by actuator index")
                            shadow.act[:] = data.act[act_idx[i]]
                        shadow.ctrl[:] = data.ctrl[act_idx[i]]
                        mujoco.mj_forward(fb_model, shadow)
                        action = controllers[i].action_for(shadow, clips[i], k)
                        data.ctrl[act_idx[i]] = action
                    for _ in range(env.n_substeps):
                        mujoco.mj_step(model, data)
                    ncontact = 0
                    total_force = 0.0
                    for ci in range(data.ncon):
                        c = data.contact[ci]
                        if {int(meta.geom_agent[c.geom1]), int(meta.geom_agent[c.geom2])} == {0, 1}:
                            ncontact += 1
                            force6 = np.zeros(6, dtype=float)
                            mujoco.mj_contactForce(model, data, ci, force6)
                            total_force += float(np.linalg.norm(force6[:3]))
                    if ncontact:
                        contacts_steps += 1
                    max_contacts = max(max_contacts, ncontact)
                    peak_pair_contact_force = max(peak_pair_contact_force, total_force)
                    zred = float(data.xpos[meta.pelvis_body_id["agent_0"], 2])
                    zblue = float(data.xpos[meta.pelvis_body_id["agent_1"], 2])
                    blue_lowest_pelvis = min(blue_lowest_pelvis, zblue)
                    blue_xy = np.asarray(data.qpos[q_idx[1]][rq:rq+2])
                    red_xy = np.asarray(data.qpos[q_idx[0]][rq:rq+2])
                    delta_blue = float(np.linalg.norm(blue_xy - initial_blue_xy))
                    delta_red = float(np.linalg.norm(red_xy - initial_red_xy))
                    max_blue_displacement = max(max_blue_displacement, delta_blue)
                    max_red_displacement = max(max_red_displacement, delta_red)
                    if k // 50 != last_report or (ncontact and contacts_steps == 1):
                        last_report = k // 50
                        print(
                            f"t={k*dt:5.2f}s contacts={ncontact:2d} "
                            f"red_pelvis={zred:.3f}m blue_pelvis={zblue:.3f}m "
                            f"red_displacement={delta_red:.3f}m "
                            f"blue_displacement={delta_blue:.3f}m "
                            f"pair_force={total_force:.1f}N"
                        )
                    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                        print("Simulation became non-finite; stopping")
                        break
                    completed_steps += 1
                    if record_states:
                        recorded.append((data.qpos.copy(), data.qvel.copy(),
                                         data.act.copy(), data.ctrl.copy()))
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0., dt - (time.perf_counter() - tick)))
                wall_elapsed = time.perf_counter() - wall_start
                simulated = completed_steps * dt
                speed = simulated / max(wall_elapsed, 1e-9)
                print(f"Simulated {simulated:.2f}s in {wall_elapsed:.2f}s wall time "
                      f"({speed:.2f}x real-time speed)")
                print(f"Pair-contact steps: {contacts_steps}; max simultaneous pair contacts: {max_contacts}")
                print(f"Peak combined inter-agent contact force: {peak_pair_contact_force:.1f} N")
                print(f"Peak root XY displacement: red={max_red_displacement:.3f}m, "
                      f"blue={max_blue_displacement:.3f}m")
                print(f"Minimum blue pelvis height: {blue_lowest_pelvis:.3f}m")
                if contacts_steps == 0:
                    print("No contact; lower --separation, or check red policy actually lunges")

            if args.replay or args.no_viewer or args.save_video is not None:
                print("Simulating without live viewer (both muscle policies remain active)...")
                simulate()
            else:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    simulate(viewer)

            if args.save_rollout is not None and recorded:
                out = args.save_rollout.expanduser().resolve()
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out,
                                    qpos=np.stack([x[0] for x in recorded]),
                                    qvel=np.stack([x[1] for x in recorded]),
                                    act=np.stack([x[2] for x in recorded]),
                                    ctrl=np.stack([x[3] for x in recorded]),
                                    frequency=np.array(1.0 / dt),
                                    red_delay=np.array(args.red_delay),
                                    separation=np.array(args.separation))
                print(f"Saved actual dynamic rollout: {out}")

            if args.save_video is not None and recorded:
                # The video is rendered from the recorded *dynamic* rollout.
                # It does not run the policies again or impose poses during simulation.
                out_video = args.save_video.expanduser().resolve()
                out_video.parent.mkdir(parents=True, exist_ok=True)
                width, height = 960, 540
                # MuJoCo's default offscreen framebuffer can be only 640x480.
                model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
                model.vis.global_.offheight = max(model.vis.global_.offheight, height)
                try:
                    import imageio.v2 as imageio
                    cam = mujoco.MjvCamera()
                    mujoco.mjv_defaultFreeCamera(model, cam)
                    cam.lookat[:] = [0.0, 0.0, 1.15]
                    cam.distance = 4.5
                    cam.azimuth = 60
                    cam.elevation = -15
                    simulated_duration = (len(recorded) - 1) * dt
                    video_frames = int(math.ceil(simulated_duration * args.video_fps)) + 1
                    print(f"Rendering {video_frames} video frames to {out_video}...")
                    with mujoco.Renderer(model, height=height, width=width) as renderer:
                        with imageio.get_writer(
                            str(out_video), fps=args.video_fps,
                            codec="libx264", quality=7, macro_block_size=2,
                        ) as writer:
                            for frame_no in range(video_frames):
                                sample_idx = min(
                                    len(recorded) - 1,
                                    round(frame_no / (args.video_fps * dt)),
                                )
                                q, v, a, ctrl = recorded[sample_idx]
                                data.qpos[:] = q
                                data.qvel[:] = v
                                data.act[:] = a
                                data.ctrl[:] = ctrl
                                data.time = sample_idx * dt
                                mujoco.mj_forward(model, data)
                                renderer.update_scene(data, camera=cam)
                                writer.append_data(renderer.render())
                    print(f"Saved MP4 video: {out_video}")
                except Exception as err:
                    print(f"Video export failed: {type(err).__name__}: {err}")
                    print("The rollout NPZ, if requested, remains saved. ")
                    print("For macOS recording, use Shift-Command-5 during viewer replay.")

            if args.replay and not args.no_viewer and recorded:
                stride = max(1, round(1.0 / (dt * args.replay_fps)))
                print(f"Replaying physical simulation at {1.0/(dt*stride):.0f} display FPS "
                      f"(original physics timestep {dt:g}s).")
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    playback_start = time.perf_counter()
                    for frame in range(0, len(recorded), stride):
                        if not viewer.is_running():
                            break
                        target_elapsed = frame * dt
                        elapsed = time.perf_counter() - playback_start
                        # If rendering falls behind, skip a frame to maintain
                        # approximately correct simulated-time playback speed.
                        if elapsed - target_elapsed > stride * dt * 1.5:
                            continue
                        sleep_for = target_elapsed - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        q, v, a, ctrl = recorded[frame]
                        data.qpos[:] = q
                        data.qvel[:] = v
                        data.act[:] = a
                        data.ctrl[:] = ctrl
                        data.time = target_elapsed
                        mujoco.mj_forward(model, data)
                        viewer.sync()
    finally:
        env.close()


if __name__ == "__main__":
    main()
