from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
import cv2
from stable_baselines3 import PPO

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    env = gym.make('myoHandWheelHoldFixed-v0')
    env.reset()

    # first policy, pushing
    pi1 = PPO.load(
        r"C:\Users\jasmi\Documents\GitHub\myosuite\MPL_baselines\policy_best_model\myoHandWheelHoldFixed-v0\2025_08_18_18_53_56_return\best_model.zip"
    )

    frames = []

    # rollout policy 1
    for _ in range(500):
        frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=0)
        frames.append(frame)
        o = env.get_obs()
        a = pi1.predict(o)[0]
        next_o, r, done, *_, ifo = env.step(
            a
        )  # take an action based on the current observation

    # insert marker frames
    for _ in range(15):  # 15 frames ~ 1.5 sec at 10 FPS
        marker = np.ones_like(frame) * 255  # white background
        cv2.putText(
            marker,
            "SWITCH TO POLICY 2",
            (40, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        frames.append(marker)

    # load second policy
    pi2 = PPO.load(
        r"C:\Users\jasmi\Documents\GitHub\myosuite\MPL_baselines\policy_best_model\myoHandWheelHoldFixed-v0\2025_08_18_17_12_25_GOOD\best_model.zip"
    )

    # rollout policy 2, starting from last state
    for _ in range(500):
        frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=0)
        frames.append(frame)
        o = env.get_obs()
        a = pi2.predict(o)[0]
        next_o, r, done, *_, ifo = env.step(
            a
        )  # take an action based on the current observation

    # save video
    skvideo.io.vwrite(
        os.path.join(curr_dir, "videos/HandFocusedRender_switch_policy.mp4"),
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p", "-r": "10"},
    )