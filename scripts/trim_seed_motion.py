"""Trim a retargeted BONES-SEED MyoFullBody trajectory by source frame indices.

Example:
    python scripts/trim_seed_motion.py --motion walk_ff_loop_270_001__A058 \
        --start 120 --end 240 --name lateral_walk_right_01

Frame ranges use [start, end), and reference the 120-Hz NPZ, not 30-FPS video frames.
Outputs are intermediate qpos-only trajectories; they are NOT complete MyoSuite MotionClips.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', required=True, help='Input NPZ filename stem')
    parser.add_argument('--start', required=True, type=int, help='Inclusive source frame index')
    parser.add_argument('--end', required=True, type=int, help='Exclusive source frame index')
    parser.add_argument('--name', required=True, help='Unique output NPZ filename stem')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = repo / 'boxing_motions' / 'seed' / 'retargeted'
    src = root / f'{args.motion}.npz'
    dst = root / f'{args.name}.npz'

    if not src.is_file():
        parser.error(f'Missing retargeted motion: {src}')
    if args.name == args.motion or dst.exists():
        parser.error(f'Output already exists or matches source: {dst}; use a unique name')

    with np.load(src, allow_pickle=False) as data:
        qpos = data['qpos']
        frequency = float(data['frequency'])
        source_file = str(data['source_file']) if 'source_file' in data.files else f'{args.motion}.bvh'

    n = len(qpos)
    if not (0 <= args.start < args.end <= n):
        parser.error(f'Expected 0 <= start < end <= {n}; got [{args.start}, {args.end})')

    clip = qpos[args.start:args.end].copy()
    np.savez_compressed(
        dst,
        qpos=clip,
        frequency=np.array(frequency),
        source_file=np.str_(source_file),
        source_frames=np.array([args.start, args.end], dtype=np.int32),
    )
    print(f'Source: {src.name} ({n} frames at {frequency:g} Hz)')
    print(f'Selected frames [{args.start}, {args.end}): {len(clip)} frames, {len(clip) / frequency:.2f} seconds')
    print(f'Saved: {dst}')
    print('Note: intermediate qpos-only file; not a full boxing_skills.py MotionClip.')


if __name__ == '__main__':
    main()
