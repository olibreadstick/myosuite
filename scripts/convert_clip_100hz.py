"""Resample the 30 Hz shadow-boxing clip to 100 Hz and rebuild all per-frame fields."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import boxing_skills as bs

MOTION_DIR = Path("boxing_motions/MyoFullBody/gmr/Transitions_mocap/mazen_c3d").resolve()
SRC = MOTION_DIR / "punch/shadow_boxing_R_001__A362.npz"
REF = MOTION_DIR / "punch/punchboxing_stand_poses.npz"
OUT = MOTION_DIR / "punch/shadow_boxing_R_001__A362_100hz.npz"
HZ = 100.0

src = np.load(SRC, allow_pickle=True)
ref = np.load(REF, allow_pickle=True)
assert [str(j) for j in src["joint_names"]] == [str(j) for j in ref["joint_names"]], \
    "joint order differs from the other clips"

q = np.asarray(src["qpos"], np.float64)
hz0 = float(src["frequency"])
for i in range(1, len(q)):                      # keep quaternion signs continuous
    if q[i, 3:7] @ q[i - 1, 3:7] < 0:
        q[i, 3:7] *= -1
t0 = np.arange(len(q)) / hz0
t1 = np.arange(int(round(t0[-1] * HZ)) + 1) / HZ
t1 = np.minimum(t1, t0[-1])
q1 = np.stack([np.interp(t1, t0, q[:, j]) for j in range(q.shape[1])], axis=1)
q1[:, 3:7] /= np.linalg.norm(q1[:, 3:7], axis=1, keepdims=True)

sites = [str(s) for s in ref["site_names"]]
fields = bs.derive_frames(np.concatenate([q1[:1], q1, q1[-1:]]), HZ, sites)

out = {k: ref[k] for k in ref.files}
for k in bs.PER_FRAME:
    out[k] = fields[k].astype(ref[k].dtype)
out["frequency"] = np.asarray(HZ, dtype=ref["frequency"].dtype)
out["split_points"] = np.array([0, len(q1)], dtype=np.int32)
np.savez(OUT, **out)
print(f"wrote {OUT.name}: {len(q1)} frames at {HZ} Hz")
