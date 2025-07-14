""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['time', 'wheel_err_right', 'hand_qpos', 'hand_qvel']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal_dist": 100.0,
        "bonus": 0.0,
        "penalty": 10,
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
        #self.object_sid = self.sim.model.site_name2id("wheel")
        self.goal_sid_right = self.sim.model.site_name2id("wheelchair_grip_right")
        self.palm_r = self.sim.model.site_name2id("palm_r")

        #self.goal_sid_left = self.sim.model.site_name2id("wheel_grip_goal_left")
        #self.object_init_pos = self.sim.data.site_xpos[self.object_sid].copy()

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        
        self.init_qpos = self.sim.model.key_qpos[0].copy() # copy the sitting + grabbing wheels keyframe


    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[13:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[12:].copy()*self.dt
        #self.obs_dict['wheel_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['wheel_err_right'] = self.sim.data.site_xpos[self.goal_sid_right] - self.sim.data.site_xpos[self.palm_r]
        #self.obs_dict['wheel_err_left'] = self.sim.data.site_xpos[self.goal_sid] - self.sim.data.site_xpos[self.object_sid]
        #self.goal_sid_right = self.sim.model.site_name2id("wheelchair_grip_right")
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[13:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[12:].copy()*self.dt
        #obs_dict['wheel_pos'] = sim.data.site_xpos[self.object_sid]
        #obs_dict['wheelchair_grip_right'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        obs_dict['wheel_err_right'] = sim.data.site_xpos[self.goal_sid_right] - sim.data.site_xpos[self.palm_r]
        #add the initial and end target points
        #could add the fingertips here,
        #obs_dict['wheel_err_left'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        dist_right = np.linalg.norm(obs_dict['wheel_err_right'])
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = dist_right > 0.300
        
        # grip_right = self._check_hand_grip_contact(
        #     hand_geom_names=["right_index_tip", "right_thumb_tip"],
        #     wheel_geom_names=[f"handrail_coll{i}" for i in range(1, 17)]
        # )

        rwd_dict = collections.OrderedDict((
            ('goal_dist', -dist_right), #exp(- k * abs(x))
            ('bonus', 1.*(dist_right<2*0) + 1.*(dist_right<0)),
            ('act_reg', -1.*act_mag),
            #('grip_bonus', 1.0 * grip_right),
            ('penalty', -1.*drop),
            ('sparse', dist_right < 0.055),
            #('sparse', 1.0 * grip_right - dist_right),
            ('solved', dist_right < 0.015),
            #('solved', grip_right and dist_right < 0.015),
            ('done', dist_right > 0.515),
        ))
        
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        
        return rwd_dict
    
    def reset(self, **kwargs):
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs

    # def _check_hand_grip_contact(self, hand_geom_names, wheel_geom_names):
    #     hand_geom_ids = [self.sim.model.geom_name2id(n) for n in hand_geom_names]
    #     wheel_geom_ids = [self.sim.model.geom_name2id(n) for n in wheel_geom_names]
        
    #     for i in range(self.sim.data.ncon):
    #         contact = self.sim.data.contact[i]
    #         if (contact.geom1 in hand_geom_ids and contact.geom2 in wheel_geom_ids) or \
    #         (contact.geom2 in hand_geom_ids and contact.geom1 in wheel_geom_ids):
    #             return True
    #     return False



# class WheelHoldRandomEnvV0(WheelHoldFixedEnvV0):

