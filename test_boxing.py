
import myosuite
from myosuite import make_env

print("Starting BoxingP0 test...")

env = make_env("myoChallengeBoxingP0-v0")

try:
    obs, info = env.reset(seed=0)

    print("Environment loaded!")
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("Simulation step successful!")
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

finally:
    env.close()
