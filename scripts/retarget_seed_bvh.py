"""Retarget one BONES-SEED SOMA BVH clip to GMR MyoFullBody joint coordinates.

Run from the MuscleMimic environment:
    uv run --extra gmr python /path/to/retarget_seed_bvh.py --motion walk_ff_loop_270_001__A058
"""
import argparse
from pathlib import Path
import numpy as np
import mujoco

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

repo = Path(__file__).resolve().parents[1]
input_dir = repo / "boxing_motions/seed/lateral_motions"
config = repo / "configs/gmr/bvh_soma_to_myofullbody.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True, help="BVH basename, with or without .bvh")
args = parser.parse_args()
motion = input_dir / (Path(args.motion).stem + ".bvh")
if not motion.is_file():
    parser.error(f"Motion not found: {motion}")
if not config.is_file():
    parser.error(f"Retarget configuration not found: {config}")

frames, height = load_bvh_file(str(motion), format="soma")
print(f"Loaded {len(frames)} frames from {motion.name}; human height: {height}")
retargeter = GeneralMotionRetargeting(
    src_human="bvh_soma", tgt_robot="myofullbody",
    actual_human_height=height, ik_config_path=str(config),
    solver="daqp", damping=0.5, verbose=False,
)
model = retargeter.model


def clamp_scalar_joints(qpos, margin=1e-5):
    qpos = qpos.copy()
    for j in range(model.njnt):
        if (model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                     mujoco.mjtJoint.mjJNT_SLIDE)
                or not model.jnt_limited[j]):
            continue
        idx = model.jnt_qposadr[j]
        low, high = model.jnt_range[j]
        inset = min(margin, max(0., (high - low) / 4))
        qpos[idx] = np.clip(qpos[idx], low + inset, high - inset)
    return qpos


retargeter.configuration.update(clamp_scalar_joints(model.qpos0))
all_qpos = []
mean_errors = []
for i, frame in enumerate(frames):
    # Keep scalar joints just inside their limits between successive IK solves.
    retargeter.configuration.update(
        clamp_scalar_joints(retargeter.configuration.data.qpos)
    )
    try:
        qpos, errors = retargeter.retarget(frame, offset_to_ground=True)
    except Exception:
        print(f"Retargeting failed at BVH frame {i} of {motion.name}")
        raise
    all_qpos.append(qpos.copy())
    mean_errors.append(float(np.mean(errors)))
    if (i + 1) % 50 == 0 or i + 1 == len(frames):
        print(f"Processed {i + 1}/{len(frames)} frames")

output_dir = repo / "boxing_motions/seed/retargeted"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / (motion.stem + ".npz")
np.savez_compressed(
    output_path,
    qpos=np.asarray(all_qpos),
    frequency=np.array(120.0),
    source_file=np.str_(motion.name),
)
print(f"Saved: {output_path}")
print(f"Mean positional error: {np.mean(mean_errors):.4f} m")
