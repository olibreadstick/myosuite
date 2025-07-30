from myosuite.utils import gym
import numpy as np
import os
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = gym.make('myoHandWheelHoldFixed-v0_left')
    env.reset()