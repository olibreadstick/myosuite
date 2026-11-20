#!/usr/bin/env python3
"""View a lunge-push motion interacting physically with a MuscleMimic defender.

Run inside myosuite-capstone with `uv run mjpython scripts/view_boxing_lunge_push.py`

Agent 0 (red): kinematically follows lunge_push.npz. It is NOT a trained
attacking policy. Agent 1 (blue): a dynamic agent controlled by a pretrained
MuscleMimic checkpoint following a standing reference.

This is a qualitative, controlled proof of physical contact and defensive
response, not a fully dynamic two-policy boxing match.
"""

import argparse
import ast
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace, ModuleType

import mujoco
import mujoco.viewer
import numpy as np

from myosuite.core.trajectory_io import load_motion_clip
from myosuite.envs.myo.tasks.challenge.boxing_vs import BoxingVsConfig, BoxingVsEnv
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyObsAdapter,
    LocalPolicyRunner,
    fullbody_history_settings_from_metadata,
    fullbody_obs_adapter_params_from_metadata,
    load_local_policy_artifacts,
    read_checkpoint_config_metadata,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MUSCLEMIMIC_REPO = REPO_ROOT.parent / "musclemimic_mcgill_dev"
MOTIONS = REPO_ROOT / "boxing_motions" / "MyoFullBody" / "gmr"
PUSH_PATH = MOTIONS / "skills" / "lunge_push.npz"
STAND_PATH = (
    MOTIONS / "Transitions_mocap" / "mazen_c3d" / "punchboxing_stand_poses.npz"
)


def quat_mul(a, b):
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(result, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return result


def quat_conj(q):
    out = np.asarray(q, dtype=np.float64).copy()
    out[1:] *= -1
    return out


def rotate(q, v):
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, np.asarray(v, dtype=float), np.asarray(q, dtype=float))
    return result


def agent_indices(model, joint_ids):
    qwidth = {0: 7, 1: 4, 2: 1, 3: 1}
    vwidth = {0: 6, 1: 3, 2: 1, 3: 1}
    # MuJoCo enum values: free=0, ball=1, slide=2, hinge=3.
    qi, vi = [], []
    for j in joint_ids:
        typ = int(model.jnt_type[j])
        qstart, vstart = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        qi.extend(range(qstart, qstart + qwidth[typ]))
        vi.extend(range(vstart, vstart + vwidth[typ]))
    return np.asarray(qi, dtype=int), np.asarray(vi, dtype=int)


def root_addresses(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if jid < 0 or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("Expected standalone MyoFullBody free joint named 'root'")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def place_clip_frame(frame, velocity, root_q, root_v, yaw, anchor, start_pos):
    """Rigidly orient a standalone clip and place its first root at anchor."""
    qyaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    pos = np.asarray(frame, dtype=float).copy()
    vel = np.asarray(velocity, dtype=float).copy()
    pos[root_q : root_q + 3] = anchor + rotate(qyaw, pos[root_q : root_q + 3] - start_pos)
    pos[root_q + 3 : root_q + 7] = quat_mul(qyaw, frame[root_q + 3 : root_q + 7])
    vel[root_v : root_v + 3] = rotate(qyaw, vel[root_v : root_v + 3])
    vel[root_v + 3 : root_v + 6] = rotate(qyaw, vel[root_v + 3 : root_v + 6])
    return pos, vel


def defender_to_local(world_q, world_v, root_q, root_v, yaw, anchor, original_pos):
    """Unrotate/uncenter world state for the standalone MuscleMimic observer."""
    inv = np.array([math.cos(yaw / 2), 0.0, 0.0, -math.sin(yaw / 2)])
    pos = world_q.copy()
    vel = world_v.copy()
    pos[root_q : root_q + 3] = original_pos + rotate(inv, pos[root_q : root_q + 3] - anchor)
    pos[root_q + 3 : root_q + 7] = quat_mul(inv, world_q[root_q + 3 : root_q + 7])
    vel[root_v : root_v + 3] = rotate(inv, vel[root_v : root_v + 3])
    vel[root_v + 3 : root_v + 6] = rotate(inv, vel[root_v + 3 : root_v + 6])
    return pos, vel


def pick_checkpoint(explicit):
    candidates = (
        [Path(explicit)] if explicit else [
            REPO_ROOT.parent / "musclemimic_mcgill_dev" / "mm-motion-100",
            REPO_ROOT / "mm-motion-100",
        ]
    )
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / "train_state").is_dir() and (path / "config" / "metadata").is_file():
            return path
    paths = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find a local mm-motion-100 checkpoint with train_state and "
        "config/metadata. Checked:\n" + paths +
        "\nRun: find .. -type d -name 'mm-motion-100' -print\n"
        "Then rerun with --checkpoint /absolute/path/to/mm-motion-100"
    )



class FullbodyTrackingObsAdapter(FullbodyObsAdapter):
    """Use the installed MuscleMimic checkout's real tracking-feature functions.

    This bridge keeps the legacy FullbodyObsAdapter for the non-goal observation
    block (joints, muscles and foot touch sensors) and replaces its goal block
    with the GoalFullBodyTracking goal prescribed by the checkpoint.  It is an
    exploratory bridge; validate observation parity against the original
    MuscleMimic environment before reporting quantitative recovery results.
    """

    def __init__(self, model, clip, metadata):
        config = metadata.get("experiment", {}).get("env_params", {})
        goal_type = config.get("goal_type")
        if goal_type not in ("GoalFullBodyTracking", "GoalFullBodyTrackingVisual"):
            raise ValueError(f"Expected GoalFullBodyTracking checkpoint, found {goal_type!r}")
        gp = fullbody_obs_adapter_params_from_metadata(metadata)
        super().__init__(model, clip, gp)
        self._tracking_lookahead = tuple(int(x) for x in gp.get("lookahead_steps", (1, 20, 40, 60, 80)))
        self._support_intent = bool(gp.get("include_support_intent", False))
        self._control_dt = float(config.get("timestep", 0.002)) * int(config.get("n_substeps", 5))
        if self._support_intent:
            raise NotImplementedError("This demo only supports include_support_intent=False")
        self._legacy_goal_dim = self.goal_dim
        self._tracking_goal_dim = None
        self._ego_root = bool(config.get("use_egocentric_root_observations", False))

        if not MUSCLEMIMIC_REPO.is_dir():
            raise FileNotFoundError(
                f"Need the original local MuscleMimic checkout: {MUSCLEMIMIC_REPO}"
            )
        # Import isolated feature helpers without initializing unrelated
        # MuscleMimic/LocoMujoco package __init__ modules. Imports of their
        # local math/egocentric helper symbols are resolved explicitly.
        def load_source_module(label: str, source: Path, *, local_modules=None):
            if not source.is_file():
                raise FileNotFoundError(f"Expected local source file: {source}")
            module = ModuleType(label)
            module.__file__ = str(source)
            sys.modules[label] = module
            try:
                if local_modules is None:
                    # math.py imports only numpy / scipy / jax, which can be
                    # imported without running loco_mujoco.__init__.
                    spec = importlib.util.spec_from_file_location(label, source)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot create import spec for {source}")
                    spec.loader.exec_module(module)
                else:
                    # tracking_features.py may use `from loco_mujoco...`
                    # or `from musclemimic...`. Keep all other imports and
                    # inject only the requested helpers from the actual
                    # original source modules, without package initialization.
                    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                    keep = []
                    for node in tree.body:
                        if isinstance(node, ast.ImportFrom) and node.module and (
                            node.module.startswith("loco_mujoco") or
                            node.module.startswith("musclemimic")
                        ):
                            dep = local_modules.get(node.module)
                            if dep is None:
                                raise ImportError(
                                    f"Unresolved standalone import: from {node.module} import "
                                    f"{', '.join(a.name for a in node.names)}"
                                )
                            for alias in node.names:
                                if alias.name == "*":
                                    raise ImportError(f"Wildcard import from {node.module} unsupported")
                                if not hasattr(dep, alias.name):
                                    raise ImportError(
                                        f"Missing helper {alias.name!r} in local module {node.module}"
                                    )
                                module.__dict__[alias.asname or alias.name] = getattr(dep, alias.name)
                            continue
                        if isinstance(node, ast.Import) and any(
                            a.name.startswith(("loco_mujoco", "musclemimic"))
                            for a in node.names
                        ):
                            raise ImportError(
                                f"Unexpected package-level import in {source.name}: "
                                f"{', '.join(a.name for a in node.names)}"
                            )
                        keep.append(node)
                    isolated = ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[]))
                    exec(compile(isolated, str(source), "exec"), module.__dict__)
            except Exception as exc:
                sys.modules.pop(label, None)
                raise ImportError(
                    f"Could not load {source.name} independently: {exc}"
                ) from exc
            return module

        # egocentric.py imports StatefulObservation from loco_mujoco, which
        # recursively initializes large parts of the MuscleMimic package.
        # Extract the original module-level helper functions, preserving
        # their implementations while avoiding unrelated package imports.
        ego_path = (
            MUSCLEMIMIC_REPO / "musclemimic/core/observations/egocentric.py"
        )
        source = ego_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(ego_path))
        wanted = {
            "rotation_matrix_from_quaternion",
            "heading_rotation_matrix",
        }
        # Include module-level helper functions as well, in case either
        # rotation function calls one of them. Do not execute the imports,
        # class declarations, or registration side effects in egocentric.py.
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        found = {node.name for node in functions}
        if not wanted.issubset(found):
            raise RuntimeError(
                f"Missing rotation helpers in {ego_path}: {sorted(wanted - found)}"
            )
        # Copy nodes before discarding decorators so the parsed AST isn't
        # reused with altered structure elsewhere.
        import copy
        functions = [copy.deepcopy(node) for node in functions]
        for node in functions:
            node.decorator_list = []
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        isolated = ast.Module(body=[future, *functions], type_ignores=[])
        isolated = ast.fix_missing_locations(isolated)
        import jax.numpy as jnp
        from scipy.spatial.transform import Rotation as np_R
        from jax.scipy.spatial.transform import Rotation as jnp_R
        # In the original source these quaternion helpers come from
        # loco_mujoco.core.utils.math.  Supplying equivalent local functions
        # prevents importing the whole loco_mujoco package (and metrx).
        def quat_scalarfirst2scalarlast(q):
            values = np.asarray(q)
            if values.shape[-1] != 4:
                raise ValueError(f"Expected quaternion shape (..., 4), got {values.shape}")
            return np.concatenate((values[..., 1:4], values[..., :1]), axis=-1)

        def quat_scalarlast2scalarfirst(q):
            values = np.asarray(q)
            if values.shape[-1] != 4:
                raise ValueError(f"Expected quaternion shape (..., 4), got {values.shape}")
            return np.concatenate((values[..., -1:], values[..., :3]), axis=-1)

        namespace = {
            "__name__": "_mm_egocentric_rotation_only",
            "quat_scalarfirst2scalarlast": quat_scalarfirst2scalarlast,
            "quat_scalarlast2scalarfirst": quat_scalarlast2scalarfirst,
            "rotation": np_R,
            "np": np,
            "jnp": jnp,
            "mujoco": mujoco,
            "math": math,
            "np_R": np_R,
            "jnp_R": jnp_R,
        }
        exec(compile(isolated, str(ego_path), "exec"), namespace, namespace)
        # tracking_features.py imports additional pure helpers (for example
        # root_velocity_in_heading_frame) from this module. Expose every
        # original top-level helper we already extracted, rather than only
        # the two used by this adapter directly. Their function globals remain
        # bound to the same prepared namespace (numpy, scipy, quaternion tools).
        self._ego = SimpleNamespace(**{name: namespace[name] for name in found})
        # Check these exact-source rotation helpers can run with numpy.
        try:
            identity = self._ego.rotation_matrix_from_quaternion(
                np.asarray([1., 0., 0., 0.]), np
            )
            heading = self._ego.heading_rotation_matrix(identity, np)
            if np.asarray(identity).shape != (3, 3) or np.asarray(heading).shape != (3, 3):
                raise ValueError("rotation/heading helper did not return 3x3 matrices")
        except Exception as exc:
            raise RuntimeError(
                f"Could not execute isolated egocentric rotation functions: {exc}. "
                "Check the original egocentric.py dependencies and confirm "
                "the quaternion/heading functions are compatible."
            ) from exc
        print("Egocentric rotation helpers loaded from original source")

        self._math = load_source_module(
            "_mm_standalone_loco_math",
            MUSCLEMIMIC_REPO / "loco_mujoco/core/utils/math.py",
        )
        # The direct-loaded tracking module uses the original helper functions
        # without importing loco_mujoco or musclemimic at package level.
        self._features = load_source_module(
            "_mm_standalone_tracking_features",
            MUSCLEMIMIC_REPO / "musclemimic/core/goals/tracking_features.py",
            local_modules={
                "loco_mujoco.core.utils.math": self._math,
                "musclemimic.core.observations.egocentric": self._ego,
            },
        )
        print("Math and tracking helpers loaded directly (no LocoMujoco initialization)")

        self._tracking_goal_dim = int(self._features.tracking_goal_dimension(
            len(self._goal.sites_for_mimic) - 1, len(self._tracking_lookahead), 0
        ))
        self.goal_dim = self._tracking_goal_dim

        # The MyoSuite clips have site_xpos/site_xmat/cvel/subtree_com.
        # GoalFullBodyTracking expects reference kinematics too. Compute them
        # from the clip qpos/qvel with MuJoCo; avoid assuming that the NPZ
        # contains optional cvel_parent/subtree_com_root arrays.
        self._ref_data = mujoco.MjData(model)
        self._tracking_cache = {}

    def _reference_frame(self, frame_idx):
        t = int(np.clip(frame_idx, 0, len(self._clip.qpos) - 1))
        cached = self._tracking_cache.get(t)
        if cached is None:
            d = self._ref_data
            d.qpos[:] = self._clip.qpos[t]
            if self._clip.qvel is None:
                d.qvel[:] = 0.0
            else:
                d.qvel[:] = self._clip.qvel[t]
            mujoco.mj_forward(self._model, d)
            # Snapshot the fields needed by the ORIGINAL calc_site_velocities
            # to avoid retaining hundreds of heavyweight MjData objects.
            cached = SimpleNamespace(
                qpos=np.asarray(d.qpos).copy(),
                qvel=np.asarray(d.qvel).copy(),
                site_xpos=np.asarray(d.site_xpos).copy(),
                site_xmat=np.asarray(d.site_xmat).copy(),
                cvel=np.asarray(d.cvel).copy(),
                subtree_com=np.asarray(d.subtree_com).copy(),
            )
            self._tracking_cache[t] = cached
        return cached

    def _tracking_goal(self, data, frame_idx):
        f = self._features
        ego = self._ego
        ref_start = self._clip.qpos[0]
        root_qidx = self._root_qpos_idx_full
        root_vidx = self._root_qvel_idx_full
        current_rq = np.asarray(data.qpos[root_qidx])
        current_pos = current_rq[:3]
        current_rot = ego.rotation_matrix_from_quaternion(current_rq[3:7], np)
        heading_rot = ego.heading_rotation_matrix(current_rot, np)
        current_vel = np.asarray(data.qvel[root_vidx])

        site_ids = self._site_ids
        site_bodyids = self._sim_site_bodyid[site_ids]
        root_bodyids = self._sim_body_rootid
        cur_sites = f.site_kinematics(data, site_ids, site_bodyids, root_bodyids, np)

        # Mirrors GoalFullBodyTracking._tracking_observation: relative XY origin
        # comes from the initial frame; current MuJoCo state is in clip-local space.
        offset = np.asarray([ref_start[root_qidx[0]], ref_start[root_qidx[1]], 0.0])
        components = []
        n = len(self._clip.qpos)
        for h, lead in enumerate(self._tracking_lookahead):
            wanted = int(frame_idx) + lead
            target_idx = max(0, min(wanted, n - 1))
            valid = float(wanted < n)
            ref = self._reference_frame(target_idx)
            ref_q = np.asarray(ref.qpos[root_qidx])
            ref_pos = ref_q[:3] - offset
            ref_rot = ego.rotation_matrix_from_quaternion(ref_q[3:7], np)
            ref_vel = np.asarray(ref.qvel[root_vidx])
            ref_sites = None
            if h == 0:
                ref_sites = f.site_kinematics(
                    ref, site_ids, site_bodyids, root_bodyids, np,
                    trajectory_site_indices=None,
                )
                components.extend([
                    f.root_tracking_error_features(
                        current_pos, current_rot, current_vel, ref_pos,
                        ref_rot, ref_vel, heading_rot, np,
                    ),
                    f.site_tracking_error_features(
                        *cur_sites, *ref_sites, heading_rot, np,
                    ),
                ])
                continue
            ref_sites_pos = f.site_positions(
                ref, site_ids, None
            )
            components.append(f.future_intent_features(
                current_pos, current_rot, ref_pos, ref_rot, ref_vel,
                ref_sites_pos, None, heading_rot,
                np.asarray(lead * self._control_dt), np.asarray(valid), np,
            ))
        goal = np.concatenate([np.asarray(part, dtype=np.float32).ravel() for part in components])
        if len(goal) != self._tracking_goal_dim:
            raise ValueError(
                f"Tracking feature dimension {len(goal)} differs from "
                f"GoalFullBodyTracking specification {self._tracking_goal_dim}"
            )
        return goal

    def build(self, data, frame_idx):
        # The parent assembles joint/muscle/touch observations followed by
        # its legacy goal. Discard the legacy goal and append the new one.
        raw_legacy = super().build(data, frame_idx)
        state = np.asarray(raw_legacy[:-self._legacy_goal_dim], dtype=np.float32)
        if self._ego_root:
            # GoalFullBodyTracking checkpoints use the MyoFullBody egocentric
            # root observation layout, not FreeJointPosNoXY + FreeJointVel:
            #   q_root_height(1), projected_gravity(3), other joint qpos,
            #   free-joint velocity in heading frame(6), other joint qvel.
            # The older adapter encodes root height + quaternion (1+4),
            # so the native layout is exactly one component shorter.
            # Reconstruct it in-place WITHOUT trimming/padding the vector.
            if not self._obs_flags["enable_joint_pos_observations"] or not self._obs_flags["enable_joint_vel_observations"]:
                raise ValueError("Egocentric root conversion requires joint position and velocity observations")
            n_nonroot_pos = len(self._qpos_non_root_ind)
            nonroot_start = 5
            root_vel_start = nonroot_start + n_nonroot_pos
            if len(state) < root_vel_start + 6:
                raise ValueError("Legacy observation unexpectedly short for egocentric root conversion")

            quat = np.asarray(data.qpos[self._root_qpos_idx_full[3:7]], dtype=np.float64)
            root_rot = np.asarray(self._ego.rotation_matrix_from_quaternion(quat, np))
            heading_rot = np.asarray(self._ego.heading_rotation_matrix(root_rot, np))
            projected_gravity = root_rot.T @ np.array([0.0, 0.0, -1.0])
            root_height = float(data.qpos[self._root_qpos_idx_full[2]])
            root_vel_heading = np.asarray(
                self._ego.root_velocity_in_heading_frame(
                    np.asarray(data.qvel[self._root_qvel_idx_full]),
                    root_rot, heading_rot, np,
                ), dtype=np.float32,
            )
            state = np.concatenate([
                np.asarray([root_height, *projected_gravity], dtype=np.float32),
                state[nonroot_start:root_vel_start],
                root_vel_heading,
                state[root_vel_start + 6:],
            ]).astype(np.float32)
        return np.concatenate([state, self._tracking_goal(data, frame_idx)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Local MuscleMimic checkpoint directory")
    ap.add_argument("--separation", type=float, default=1.0, help="Root-to-root separation, metres")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=float, default=0.4, help="Standing time before lunge, seconds")
    args = ap.parse_args()

    if not PUSH_PATH.is_file() or not STAND_PATH.is_file():
        raise FileNotFoundError(f"Missing motion: {PUSH_PATH} or {STAND_PATH}")
    ckpt_path = pick_checkpoint(args.checkpoint)

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

        fb_cfg = default_mimic_fullbody_config()
        fb_cfg.disable_fingers = True
        fb_model, _, _ = compile_mimic_fullbody_mjmodel(fb_cfg)
        fb_data = mujoco.MjData(fb_model)
        r_q, r_v = root_addresses(fb_model)

        attacker = load_motion_clip(PUSH_PATH, fb_model.nq, fb_model.nv)
        standing = load_motion_clip(STAND_PATH, fb_model.nq, fb_model.nv)
        if attacker.qvel is None or standing.qvel is None:
            raise ValueError("Both motion clips must include qvel")

        qa, va = agent_indices(model, meta.jnt_ids["agent_0"])
        qd, vd = agent_indices(model, meta.jnt_ids["agent_1"])
        if any((len(qa) != fb_model.nq, len(qd) != fb_model.nq,
                len(va) != fb_model.nv, len(vd) != fb_model.nv)):
            raise ValueError(
                f"Combined model agent sizes {len(qa)}, {len(va)} do not "
                f"match standalone {fb_model.nq}, {fb_model.nv}. "
                "Check disable_fingers=True and model version."
            )

        print(f"Model: combined nq={model.nq}, nu={model.nu}; "
              f"single nq={fb_model.nq}, nu={fb_model.nu}")
        print(f"Lunge clip: {len(attacker.qpos)} frames at "
              f"{attacker.frequency_hz} Hz")

        metadata = read_checkpoint_config_metadata(ckpt_path)
        artifacts = load_local_policy_artifacts(ckpt_path)
        if artifacts.action_dim != len(meta.act_indices["agent_1"]):
            raise ValueError(
                f"Checkpoint action dim={artifacts.action_dim}, "
                f"defender actuators={len(meta.act_indices['agent_1'])}"
            )
        adapter = FullbodyTrackingObsAdapter(fb_model, standing, metadata)
        if adapter._ego_root:
            print("Using checkpoint egocentric root state: height + projected gravity "
                  "+ heading-frame velocity")
        print("NOTE: reference goal and root encoding are reconstructed for this "
              "exploratory viewer; full observation parity is not yet validated.")
        test_data = mujoco.MjData(fb_model)
        test_data.qpos[:] = standing.qpos[0]
        test_data.qvel[:] = standing.qvel[0]
        mujoco.mj_forward(fb_model, test_data)
        sample_obs = adapter.build(test_data, 0)
        print(f"Observation: {len(sample_obs)} (checkpoint expects {artifacts.obs_dim}); "
              f"state={len(sample_obs)-adapter.goal_dim}, "
              f"tracking goal={adapter.goal_dim}")
        if len(sample_obs) != artifacts.obs_dim:
            raise ValueError(
                "Goal tracking observation still differs from checkpoint; "
                "stop rather than passing invalid inputs to the policy."
            )
        policy = LocalPolicyRunner(
            artifacts=artifacts,
            stochastic=False,
            seed=0,
            obs_adapter=adapter,
            **fullbody_history_settings_from_metadata(metadata),
        )

        # Attacker is in front (+Y), defender behind (-Y), facing attacker.
        attack_src0 = attacker.qpos[0, r_q : r_q + 3].copy()
        displacements = attacker.qpos[:, r_q : r_q + 2] - attack_src0[:2]
        primary = displacements[np.argmax(np.linalg.norm(displacements, axis=1))]
        if np.linalg.norm(primary) > 0.05:
            attacker_yaw = -np.pi / 2 - math.atan2(primary[1], primary[0])
        else:
            attacker_yaw = 0.0
        print(f"Lunge root displacement: {displacements[-1].round(3)} m; "
              f"max: {np.linalg.norm(primary):.3f} m")
        print(f"Attacker yaw correction: {np.degrees(attacker_yaw):.1f} degrees")

        attack_anchor = np.array([0.0, args.separation / 2, attack_src0[2]])
        def_src0 = standing.qpos[0, r_q : r_q + 3].copy()
        def_anchor = np.array([0.0, -args.separation / 2, def_src0[2]])
        defender_yaw = np.pi
        pose, vel = place_clip_frame(
            standing.qpos[0], standing.qvel[0], r_q, r_v,
            defender_yaw, def_anchor, def_src0,
        )
        data.qpos[qd], data.qvel[vd] = pose, vel
        mujoco.mj_forward(model, data)

        dt = cfg.ctrl_dt
        n_steps = max(1, int(args.seconds / dt))
        warmup_steps = int(args.warmup / dt)
        contacts_seen = 0
        peak_contacts = 0
        min_pelvis = float('inf')

        print("Red = reference-motion-driven lunge (kinematically enforced)")
        print("Blue = fully dynamic pretrained MuscleMimic standing controller")
        print("Model contact is ON; episode will not reset after a fall")
        print("Close viewer or press ESC to exit")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for k in range(n_steps):
                if not viewer.is_running():
                    break
                tic = time.perf_counter()
                frame_id = min(max(0, k - warmup_steps), len(attacker.qpos) - 1)
                atk_q, atk_v = place_clip_frame(
                    attacker.qpos[frame_id], attacker.qvel[frame_id],
                    r_q, r_v, attacker_yaw, attack_anchor, attack_src0,
                )
                if k < warmup_steps or k - warmup_steps >= len(attacker.qpos):
                    atk_v[:] = 0.0
                data.qpos[qa], data.qvel[va] = atk_q, atk_v
                # Attacker uses no learned muscle command; the motion is imposed.
                data.ctrl[meta.act_indices["agent_0"]] = 0.0
                mujoco.mj_forward(model, data)

                # Convert defender's combined-world state to the canonical
                # standalone model frame expected by the MuscleMimic policy.
                defender_local_q, defender_local_v = defender_to_local(
                    data.qpos[qd], data.qvel[vd], r_q, r_v,
                    defender_yaw, def_anchor, def_src0,
                )
                fb_data.qpos[:] = defender_local_q
                fb_data.qvel[:] = defender_local_v
                if fb_model.na:
                    aidx = np.asarray(meta.act_indices["agent_1"], dtype=int)
                    fb_data.act[:] = data.act[aidx]
                fb_data.ctrl[:] = data.ctrl[meta.act_indices["agent_1"]]
                mujoco.mj_forward(fb_model, fb_data)
                stand_frame = min(k, len(standing.qpos) - 1)
                action = policy.action_for(fb_data, standing, stand_frame)
                data.ctrl[meta.act_indices["agent_1"]] = action

                for _ in range(env.n_substeps):
                    mujoco.mj_step(model, data)

                current_contacts = 0
                for j in range(data.ncon):
                    c = data.contact[j]
                    owner1 = int(meta.geom_agent[c.geom1])
                    owner2 = int(meta.geom_agent[c.geom2])
                    if {owner1, owner2} == {0, 1}:
                        current_contacts += 1
                if current_contacts:
                    contacts_seen += 1
                    peak_contacts = max(peak_contacts, current_contacts)
                pelvis_z = float(data.xpos[meta.pelvis_body_id["agent_1"], 2])
                min_pelvis = min(min_pelvis, pelvis_z)
                if k % 50 == 0 or (current_contacts and contacts_seen == 1):
                    print(f"t={k*dt:5.2f}s  contacts={current_contacts}  "
                          f"defender_pelvis_z={pelvis_z:.3f}m")
                if not np.isfinite(data.qpos).all():
                    print("Physics became non-finite; stopping")
                    break
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - tic)))
        print(f"Contact steps: {contacts_seen}; max simultaneous contacts: {peak_contacts}")
        print(f"Minimum defender pelvis height: {min_pelvis:.3f}m")
        if not contacts_seen:
            print("No agent-to-agent contact detected: try --separation 0.8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
