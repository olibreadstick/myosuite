
from myosuite import make_env

print("Starting Chase Tag test...")

env = make_env("myoChallengeChaseTagFBVs-v0")

try:
    obs, info = env.reset(seed=0)

    print("Environment loaded!")
    print("Agents:", list(obs.keys()))

    for agent_id in obs:
        print(f"{agent_id} observation shape:", obs[agent_id].shape)
        print(f"{agent_id} action space:", env.action_space[agent_id])

    actions = {
        agent_id: env.action_space[agent_id].sample()
        for agent_id in obs
    }

    obs, rewards, terminated, truncated, info = env.step(actions)

    print("Simulation step successful!")
    print("Rewards:", rewards)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

finally:
    env.close()
