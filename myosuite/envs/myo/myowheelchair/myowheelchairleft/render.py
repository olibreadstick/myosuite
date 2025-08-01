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
    pi = PPO.load(r"/Users/oliviacardillo/myosuite/myosuite3/MPL_baselines/policy_best_model/myoHandWheelHoldFixed-v0/2025_07_31_16_46_23/best_model.zip")

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
        curr_dir+"/videos/HandFocusedRender_traj_olivia_from_2025_07_31_16_46_23.mp4",
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p", "-r": "10"},
    )