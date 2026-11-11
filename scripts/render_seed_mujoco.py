
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

from general_motion_retargeting.motion_retarget import ROBOT_XML_DICT

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--motion",
    required=True,
    help="Motion filename without .npz"
)
args = parser.parse_args()

repo = Path(__file__).resolve().parents[1]

motion_path = (
    repo
    / "boxing_motions/seed/retargeted"
    / f"{args.motion}.npz"
)

if not motion_path.exists():
    raise FileNotFoundError(f"Motion not found: {motion_path}")

output_dir = repo / "logs/videos/seed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{args.motion}_mujoco.mp4"

print("Loading:", motion_path)
print("Saving video to:", output_path)

# Load the MyoFullBody model used for retargeting
model = mujoco.MjModel.from_xml_path(
    str(ROBOT_XML_DICT["myofullbody"])
)
data = mujoco.MjData(model)

# Load retargeted trajectory
with np.load(motion_path) as motion:
    trajectory = motion["qpos"]
    frequency = float(motion["frequency"])

print("Model qpos dimensions:", model.nq)
print("Motion dimensions:", trajectory.shape)

if trajectory.shape[1] != model.nq:
    raise ValueError("Motion and MuJoCo model dimensions do not match.")

# Render at 30 FPS rather than all 120 source frames per second
output_fps = 30
frame_step = max(1, round(frequency / output_fps))

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 3.8
camera.azimuth = 90
camera.elevation = 5

print(f"Rendering {len(trajectory)} source frames...")

with mujoco.Renderer(model, height=480, width=640) as renderer:
    with imageio.get_writer(
        str(output_path),
        fps=frequency / frame_step,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:

        for i in range(0, len(trajectory), frame_step):
            data.qpos[:] = trajectory[i]
            mujoco.mj_forward(model, data)

            # Follow the humanoid during lateral movement
            camera.lookat[:] = data.qpos[:3] + np.array([0, 0, 0.15])

            renderer.update_scene(data, camera)
            frame = renderer.render()

            writer.append_data(frame)

            if i % 120 == 0:
                print(f"Rendered source frame {i}/{len(trajectory)}")

print("\nVideo saved to:", output_path)
