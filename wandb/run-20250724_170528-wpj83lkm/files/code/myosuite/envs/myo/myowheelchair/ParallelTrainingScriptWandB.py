import myosuite
from myosuite.utils import gym
import numpy as np
import os
import torch
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import skvideo.io
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_name", type=str, default='myoStandingBack-v0', help="environment name")
parser.add_argument("--group", type=str, default='testing', help="group name")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")
parser.add_argument("--algo", type=str, default='PPO', help="algorithm for training")


args = parser.parse_args()


def make_env(env_name, idx, seed=0):
    def _init():
        env = gym.make(env_name)
        env.seed(seed + idx)
        log_dir = f"./logs/{env_name}_{idx}"
        env = Monitor(env, filename=log_dir)
        return env
    return _init

curr_dir = os.path.dirname(os.path.abspath(__file__))

def record_video(env_name, log_path):
    env = gym.make('myoHandWheelHoldFixed-v0')
    env.reset()

    # model = PPO("MlpPolicy", env, verbose=0)
    model_path = log_path

    pi = PPO.load(model_path)

    # render
    frames = []
    for _ in range(500):
        frames.append(env.sim.renderer.render_offscreen(width=400, height=400, camera_id=0)) 
        o = env.get_obs()
        a = pi.predict(o)[0]
        next_o, r, done, *_, ifo = env.step(
            a
        )  # take an action based on the current observation

    # make a local copy
    skvideo.io.vwrite(
        curr_dir+"/videos/Curr_Best.mp4",
        np.asarray(frames),
        outputdict={"-pix_fmt": "yuv420p", "-r": "10"},
    )

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.training_env.get_obs_vec()
        self.logger.record("obs", value)

        return True


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    num_cpu = 12

    dof_env = ['myoHandWheelHoldFixed-v0']

    training_steps = 5e5
    #wandb
    for env_name in dof_env:
        print('Begin training')
        ENTROPY = 0.01
        start_time = time.time()
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        time_now = time_now + str(args.seed) + args.algo
        print(time_now + '\n\n')
        LR = linear_schedule(args.learning_rate)
        CR = linear_schedule(args.clip_range)

        IS_WnB_enabled = False


    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        IS_WnB_enabled = True
        config = {
            "policy_type": 'PPO',
            'name': time_now,
            "total_timesteps": training_steps,
            "env_name": env_name,
            "dense_units": 512,
            "activation": "relu",
            "max_episode_steps": 200,
            "seed": args.seed,
            "entropy": ENTROPY,
            "lr": args.learning_rate,
            "CR": args.clip_range,
            "num_envs": args.num_envs,
            "loaded_model": 'NA',
        }
        #config = {**config, **envs.rwd_keys_wt}
        run = wandb.init(
                        project="myoHandWheelHoldFixed-v0",
                        entity="oliviacardillo-mcgill-university",
                        group=args.group,
                        settings=wandb.Settings(start_method="thread"),
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        monitor_gym=True,  # auto-upload the videos of agents playing the game
                        save_code=True,  # optional
                        )
    except ImportError as e:
        pass 



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

    model = PPO('MlpPolicy', envs, verbose=1, ent_coef=0.001,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"runs/{time_now}")
    
    # TODO TRY LOADING
    #model_num =   '2025_07_23_19_37_27'
    #model = PPO.load('./MPL_baselines/policy_best_model'+ '/'+ env_name + '/' + model_num + r'/best_model', envs, verbose = 1, ent_coeff = 0.01, policy_kwargs = policy_kwargs, tensorboard_log=f"runs/{time_now}")

    obs_callback = TensorboardCallback()
    callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=100)])#, obs_callback])

    #TODO TOTAL TIMESTEPS HERE
    model.learn(total_timesteps=5e5, tb_log_name=env_name + "_" + time_now, callback=callback)
    model.save(curr_dir+'/WheelDist_policy')

    # evaluate policy
    all_rewards = []
    ep_rewards = []
    done = False
    obs = envs.reset()
    done = False
    for _ in range(100):
        # get the next action from the policy
        action, _ = model.predict(obs, deterministic=True)
        # take an action based on the current observation
        obs, reward, done, info = envs.step(action)
        ep_rewards.append(reward)
    all_rewards.append(np.sum(ep_rewards))
    print("All episode rewards:", all_rewards)
    print(f"Average reward: {np.mean(all_rewards)} over 5 episodes")
    all_rewards

    # # Record video after training
    record_video(env_name, log_path)