from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
from stable_baselines3 import PPO
import myosuite.envs.myo.myowheelchair.myowheelchairleft.__init__


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))


    env = gym.make('myoHandWheelHoldFixed-v0_left')
    env.reset()


    model = PPO("MlpPolicy", env, verbose=0)
    pi = PPO.load(r"C:\Users\jasmi\Documents\GitHub\myosuite\MPL_baselines_left\policy_best_model_left\myoHandWheelHoldFixed-v0_left\2025_07_30_13_10_41\best_model.zip")

    # render
    frames = []
    for _ in range(800):
        frames.append(env.sim.renderer.render_offscreen(width=400, height=400, camera_id=1)) 
        o = env.get_obs()
        a = pi.predict(o)[0]
        next_o, r, done, *_, ifo = env.step(
            a
        )  # take an action based on the current observation

    # make a local copy
    skvideo.io.vwrite(
        curr_dir+"/videos/HandFocusedRender_4.mp4",
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p", "-r": "10"},
    )