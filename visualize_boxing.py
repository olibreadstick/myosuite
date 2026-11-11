
import time

import mujoco.viewer
from myosuite import make_env

env = make_env("myoChallengeBoxingP0-v0")
obs, info = env.reset(seed=0)

print("BoxingP0 loaded!")
print("Opening MuJoCo viewer...")

try:
    with mujoco.viewer.launch_passive(
        env.unwrapped.model,
        env.unwrapped.data,
    ) as viewer:

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

finally:
    env.close()
