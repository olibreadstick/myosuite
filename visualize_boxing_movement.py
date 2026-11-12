# view_clip.py   usage: mjpython view_clip.py clip.npz [body_index]
import sys, time
import numpy as np
import mujoco, mujoco.viewer
from myosuite import make_env

path = sys.argv[1]
body_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 or 1: which copy to drive

d = np.load(path, allow_pickle=True)
q = d["qpos"].astype(float)
fps = float(d["frequency"]) if "frequency" in d.files else 30.0
names = [str(n) for n in d["joint_names"]]
types = d["jnt_type"].astype(int)          # 0=free, 1=ball, 2=slide, 3=hinge
assert len(names) == len(types)

# qpos column offset of each clip joint
widths = [7 if t == 0 else 4 if t == 1 else 1 for t in types]
offsets = np.concatenate([[0], np.cumsum(widths)])
assert offsets[-1] == q.shape[1], f"{offsets[-1]} cols expected, clip has {q.shape[1]}"

env = make_env("myoChallengeBoxingVs-v0")
env.reset(seed=0)
model, data = env.unwrapped.model, env.unwrapped.data
env_names = [model.joint(i).name for i in range(model.njnt)]

# find the prefixes the env adds to the clip's joint names
probe = next(n for n in names if n and not n.lower().startswith("root"))
prefixes = [en[:-len(probe)] for en in env_names if en.endswith(probe)]
print("prefixes found:", prefixes)
prefix = prefixes[body_idx]
print(f"driving body {body_idx} with prefix '{prefix}'")

src, dst = [], []   # clip column -> env qpos address
for n, t, off in zip(names, types, offsets[:-1]):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
    if j < 0:
        continue
    for k in range(widths[list(names).index(n)]):
        src.append(off + k)
        dst.append(model.jnt_qposadr[j] + k)
print(f"matched {len(set(dst))} qpos entries (clip has {q.shape[1]})")
src, dst = np.array(src), np.array(dst)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i, row in enumerate(q):
                if not viewer.is_running():
                    break
                data.qpos[dst] = row[src]
                mujoco.mj_forward(model, data)
                viewer.sync()
                if i % 5 == 0: print(f"frame {i}  t={i/fps:.2f}s", flush=True)
                time.sleep(1.0 / fps)
finally:
    env.close()