from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
from stable_baselines3 import PPO

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    env = gym.make('myoHandWheelHoldFixed-v0')
    env.reset()

    model = PPO("MlpPolicy", env, verbose=0)
    pi = PPO.load(curr_dir+"/WheelDist_policy")

    env = gym.make('myoHandWheelHoldFixed-v0')
    env.reset()

    # render
    frames = []
    for _ in range(300):
        frames.append(env.sim.renderer.render_offscreen(width=400, height=400, camera_id=4)) 
        o = env.get_obs()
        a = pi.predict(o)[0]
        next_o, r, done, *_, ifo = env.step(
            a
        )  # take an action based on the current observation

    # make a local copy
    skvideo.io.vwrite(
        curr_dir+"/videos/HandFocusedRender.mp4",
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p", "-r": "10"},
    )