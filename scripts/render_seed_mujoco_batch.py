"""Render a retargeted BONES-SEED MyoFullBody qpos trajectory to MP4."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from general_motion_retargeting.motion_retarget import ROBOT_XML_DICT

repo = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True, help="Motion basename (without .npz)")
args = parser.parse_args()
name = Path(args.motion).stem
motion_path = repo / "boxing_motions/seed/retargeted" / (name + ".npz")
if not motion_path.is_file():
    parser.error(f"Retargeted motion not found: {motion_path}")
output_dir = repo / "logs/videos/seed"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / (name + "_mujoco.mp4")

model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["myofullbody"]))
data = mujoco.MjData(model)
with np.load(motion_path) as source:
    trajectory = source["qpos"]
    frequency = float(source["frequency"])
if trajectory.ndim != 2 or trajectory.shape[1] != model.nq:
    raise ValueError(f"Expected trajectory shape (frames, {model.nq}), got {trajectory.shape}")

output_fps = 30
frame_step = max(1, round(frequency / output_fps))
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 3.8
camera.azimuth = 90
camera.elevation = 5

print(f"Rendering {name}: {len(trajectory)} frames at source {frequency:g} Hz")
with mujoco.Renderer(model, height=480, width=640) as renderer:
    with imageio.get_writer(
        str(output_path), fps=frequency / frame_step,
        codec="libx264", quality=8, macro_block_size=16,
    ) as writer:
        for i in range(0, len(trajectory), frame_step):
            data.qpos[:] = trajectory[i]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.array([0, 0, 0.15])
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())
print(f"Saved: {output_path}")
