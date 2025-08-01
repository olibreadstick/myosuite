""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

# Trying trajectory approach

import collections
import numpy as np
import math
from myosuite.utils import gym
import mujoco
import pandas as pd
import os


from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0Left(BaseV0):

    DEFAULT_OBS_KEYS = ['time', 'hand_qpos', 'hand_qvel']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "trajectory_rwd": 5.0,
        "bonus": 2.0,
        "penalty": 5.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)


    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):

        self.palm_r = self.sim.model.site_name2id("palm_r")
        self.target_qpos = self.sim.model.key_qpos[0].copy() # hand closing in on rail

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.return_traj = pd.read_csv(curr_dir + '/Return_Phase_Trajectory.csv').values
        self.return_start_time = self.sim.data.time
        self.return_duration = 1.5  # seconds

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos = self.sim.model.key_qpos[1].copy() # copy returning keyframe
        
    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[13:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[12:].copy()*self.dt

        self.obs_dict["palm_pos"] = self.sim.data.site_xpos[self.palm_r]

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[13:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[12:].copy()*self.dt
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_r]

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        step = int((self.sim.data.time - self.return_start_time) / self.return_duration * len(self.return_traj))
        step = np.clip(step, 0, len(self.return_traj) - 1)
        target_qpos = self.return_traj[step]
        current_qpos = self.sim.data.qpos.copy()



        traj_err = np.linalg.norm(current_qpos - target_qpos) / len(current_qpos)
        
        rwd_dict = collections.OrderedDict((
            # Optional shaping reward
            ('trajectory_rwd', -traj_err),

            # MUST KEYS
            ('bonus', 1.0 * (traj_err < 0.2) + 1.0 * (traj_err < 0.1)),
            ('penalty', -1.0 * (traj_err > 1.5)),  # drop if far off path
            ('sparse', -10.0 * traj_err),
            ('solved', float(traj_err < 0.0025)),
            ('done', bool(traj_err > 1.5)),
        ))

        
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        # self.prev_rwd_dict = rwd_dict  # Always update
        return rwd_dict
    
    def reset(self, **kwargs):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.return_traj = pd.read_csv(curr_dir + '/Return_Phase_Trajectory.csv').values
        self.return_start_time = self.sim.data.time
        self.return_duration = 1.0  # seconds

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)

        self.sim.data.qpos[:] = self.sim.model.key_qpos[1]
        self.sim.data.qvel[:] = 0

        return obs