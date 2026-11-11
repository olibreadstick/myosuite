"""Render a BONES-SEED BVH motion as an MP4 stick-figure animation.

Usage:
    uv run python scripts/visualize_seed_bvh.py boxing_motions/seed/lateral_motions/lateral_speed_step_ff_270_R_003__A359.bvh

The script preserves the BVH joint-channel order and displays Y as vertical.
This is reference-motion visualization, not MyoSuite retargeting.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation


def load_positions(path, stride=4):
    mocap = Bvh(path.read_text())
    joints = mocap.get_joints_names()
    parents = [mocap.joint_parent_index(j) for j in joints]
    offsets = np.array([mocap.joint_offset(j) for j in joints]) / 100.0  # SOMA offsets are cm
    channels = [mocap.joint_channels(j) for j in joints]
    starts = np.cumsum([0] + [len(ch) for ch in channels[:-1]])
    frames = np.asarray(mocap.frames, dtype=float)
    if frames.shape[1] != sum(map(len, channels)):
        raise ValueError("Unexpected BVH frame channel count")
    indices = np.arange(0, len(frames), stride)
    positions = np.zeros((len(indices), len(joints), 3))

    for out_i, frame_i in enumerate(indices):
        frame = frames[frame_i]
        rotations = [Rotation.identity() for _ in joints]
        for j, (parent, start, ch) in enumerate(zip(parents, starts, channels)):
            values = frame[start:start + len(ch)]
            translation = offsets[j].copy()
            rotation_axes, rotation_values = [], []
            for name, value in zip(ch, values):
                if name.endswith("position"):
                    translation["XYZ".index(name[0])] += value / 100.0
                elif name.endswith("rotation"):
                    rotation_axes.append(name[0]); rotation_values.append(value)
            local_rotation = (
                Rotation.from_euler("".join(rotation_axes), rotation_values, degrees=True)
                if rotation_axes else Rotation.identity()
            )
            if parent < 0:
                positions[out_i, j] = translation
                rotations[j] = local_rotation
            else:
                positions[out_i, j] = positions[out_i, parent] + rotations[parent].apply(translation)
                rotations[j] = rotations[parent] * local_rotation
    return positions, parents, indices, mocap.frame_time


def render(path, output, stride=4, fps=30):
    positions, parents, indices, frame_time = load_positions(path, stride)
    # Render only the limb/trunk skeleton; omit finger bones for visual clarity.
    all_xyz = positions.reshape(-1, 3)
    lows, highs = np.percentile(all_xyz, [1, 99], axis=0)
    center = (lows + highs) / 2
    radius = max(np.max(highs - lows) * 0.62, 0.9)
    fig = plt.figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y up (m)")
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[2]-radius, center[2]+radius)
    ax.set_zlim(max(0, center[1]-radius), center[1]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=17, azim=-65)
    lines = [ax.plot([], [], [], lw=2)[0] for p in parents if p >= 0]
    edges = [(j, p) for j, p in enumerate(parents) if p >= 0]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Original is 120 Hz. 4x downsampling gives approximately 30 fps.
    playback_fps = int(round(1.0 / (frame_time * stride)))
    with imageio.get_writer(output, fps=min(fps, playback_fps), codec="libx264", macro_block_size=1) as writer:
        for i, original_frame in enumerate(indices):
            points = positions[i]
            for line, (j, p) in zip(lines, edges):
                a, b = points[[p, j]]
                line.set_data([a[0], b[0]], [a[2], b[2]])
                line.set_3d_properties([a[1], b[1]])
            ax.set_title(f"{path.stem} | BVH frame {original_frame} | {original_frame*frame_time:.2f}s")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
    plt.close(fig)
    print(f"Saved {output} ({len(indices)} frames at {playback_fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    if not args.bvh.is_file():
        parser.error(f"BVH not found: {args.bvh}")
    output = args.output or Path("logs/videos/seed") / (args.bvh.stem + ".mp4")
    render(args.bvh, output, stride=args.stride)
