""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant

import os
import numpy as np


# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )

curr_dir = os.path.dirname(os.path.abspath(__file__))

print("MyoSuite:> Registering Myo Envs")

# WheelHold ==============================
register_env_with_variants(id='myoHandWheelHoldFixed-v0',
        entry_point='myosuite.envs.myo.myowheelchair.wheelhold_v0:WheelHoldFixedEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/wheelchair/myowc_skeleton.xml',
            'normalize_act': True
        }
    )



