import myosuite
from myosuite.utils import gym
import numpy as np
import os
import torch
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import skvideo.io

def make_env(env_name, idx, seed=0):
    def _init():
        env = gym.make(env_name)
        env.seed(seed + idx)
        return env
    return _init

def record_video(env_name, model_path="WheelDist_policy", out_path="videos/RockPose.mp4"):
    env = gym.make(env_name)
    env.reset()
    model = PPO.load(model_path)
    frames = []
    for _ in range(300):
        frames.append(env.sim.renderer.render_offscreen(width=400, height=400, camera_id=0))
        obs = env.get_obs()
        action, _ = model.predict(obs)
        env.step(action)
    os.makedirs("videos", exist_ok=True)
    skvideo.io.vwrite(out_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p", "-r": "10"})

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    num_cpu = 2
    env_name = 'myoHandWheelHoldFixed-v0'
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = f'./MPL_baselines/policy_best_model/{env_name}/{time_now}/'

    envs = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])

    eval_callback = EvalCallback(envs, best_model_save_path=log_path, log_path=log_path,
                                 eval_freq=10000, deterministic=True, render=False)

    policy_kwargs = {
        'activation_fn': torch.nn.ReLU,
        'net_arch': {'pi': [256, 256], 'vf': [256, 256]}
    }

    print("Begin training")
    model = PPO('MlpPolicy', envs, verbose=1, ent_coef=0.001, policy_kwargs=policy_kwargs)
    callback = CallbackList([eval_callback])
    model.learn(total_timesteps=5000000, tb_log_name=env_name + "_" + time_now, callback=callback)
    model.save("WheelDist_policy")

    # Record video after training
    record_video(env_name)
