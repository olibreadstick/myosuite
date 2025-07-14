import myosuite
from myosuite.utils import gym
import numpy as np
import os
import torch
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallBack
from wandb.integration.sb3 import WandbCallback
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
    multiprocessing.set_start_method("fork", force=True)

    num_cpu = 8


    #wandb

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
        run = wandb.init(project="MyoBack_Train",
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
    model = PPO('MlpPolicy', envs, verbose=1, ent_coef=0.001, policy_kwargs=policy_kwargs)
    obs_callback = TensorboardCallback()
    callback = CallbackList([eval_callback, WandbCallback(gradient_save_freq=100)])#, obs_callback])
    model.learn(total_timesteps=1000000, tb_log_name=env_name + "_" + time_now, callback=callback)
    model.save("WheelDist_policy")


    # Record video after training
    record_video(env_name)

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