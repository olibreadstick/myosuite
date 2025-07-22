""" =================================================
# Copyleft (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import math
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0Left(BaseV0):

    DEFAULT_OBS_KEYS = ['time', 'wheel_err_left', 'hand_qpos', 'hand_qvel']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal_dist": 10.0,
        "hand_dist" : 5.0,
        "fin_open": -10.0,
        "pinky_close" :-15.0,
        "bonus": 0.0,
        "penalty": 2,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__left(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__left(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)


    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.goal_sid_left = self.sim.model.site_name2id("wheelchair_grip_left")
        self.palm_l = self.sim.model.site_name2id("palm_l")
        self.hand_start_left = self.sim.model.site_name2id("hand_start_left")
        self.rail_bottom_left = self.sim.model.site_name2id("rail_bottom_left")

        # define the palm and tip site id.
        # self.palm_l = self.sim.model.site_name2id('S_grasp')
        self.init_palm_z = self.sim.data.site_xpos[self.palm_l][-1]
        self.fin0 = self.sim.model.site_name2id("THtip")
        self.fin1 = self.sim.model.site_name2id("IFtip")
        self.fin2 = self.sim.model.site_name2id("MFtip")
        self.fin3 = self.sim.model.site_name2id("RFtip")
        self.fin4 = self.sim.model.site_name2id("LFtip")

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
        self.obs_dict['wheel_err_left'] = self.sim.data.site_xpos[self.goal_sid_left] - self.sim.data.site_xpos[self.palm_l]
        self.obs_dict['hand_initpos_err_left'] = self.sim.data.site_xpos[self.hand_start_left]- self.sim.data.site_xpos[self.goal_sid_left]

        self.obs_dict["palm_pos"] = self.sim.data.site_xpos[self.palm_l]
        self.obs_dict['fin0'] = self.sim.data.site_xpos[self.fin0]
        self.obs_dict['fin1'] = self.sim.data.site_xpos[self.fin1]
        self.obs_dict['fin2'] = self.sim.data.site_xpos[self.fin2]
        self.obs_dict['fin3'] = self.sim.data.site_xpos[self.fin3]
        self.obs_dict['fin4'] = self.sim.data.site_xpos[self.fin4]

        self.obs_dict["rail_bottom_left"] = self.sim.data.site_xpos[self.rail_bottom_left]

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
        #obs_dict['wheelchair_grip_left'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        obs_dict['wheel_err_left'] = sim.data.site_xpos[self.goal_sid_left] - sim.data.site_xpos[self.palm_l]
        obs_dict['hand_initpos_err_left'] = sim.data.site_xpos[self.hand_start_left]- sim.data.site_xpos[self.goal_sid_left]
        #add the initial and end target points
        #could add the fingertips here,
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_l]
        obs_dict['fin0'] = sim.data.site_xpos[self.fin0]
        obs_dict['fin1'] = sim.data.site_xpos[self.fin1]
        obs_dict['fin2'] = sim.data.site_xpos[self.fin2]
        obs_dict['fin3'] = sim.data.site_xpos[self.fin3]
        obs_dict['fin4'] = sim.data.site_xpos[self.fin4]

        obs_dict["rail_bottom_left"] = sim.data.site_xpos[self.rail_bottom_left]

        #obs_dict['wheel_err_left'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        dist_left = np.linalg.norm(obs_dict['wheel_err_left'])
        hand_initpos_err_left = np.linalg.norm(obs_dict['hand_initpos_err_left'])
        
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = dist_left > 0.500

        fin_keys = ['fin0', 'fin1', 'fin2', 'fin3', 'fin4']
        # for fin in fin_keys:
        #     print(fin, type(obs_dict[fin]), np.shape(obs_dict[fin]))
        fin_open = sum(
            np.linalg.norm(obs_dict[fin].squeeze() - obs_dict['rail_bottom_left'].squeeze(), axis=-1)
            for fin in fin_keys
        )
        pinky_close = np.linalg.norm(obs_dict['fin4'].squeeze() - obs_dict['rail_bottom_left'].squeeze(), axis=-1)
        
        # grip_left = self._check_hand_grip_contact(
        #     hand_geom_names=["left_index_tip", "left_thumb_tip"],
        #     wheel_geom_names=[f"handrail_coll{i}" for i in range(1, 17)]
        # )

        rwd_dict = collections.OrderedDict((
            ('goal_dist', math.exp(-2.0*abs(dist_left))), #exp(- k * abs(x))
            ('hand_dist', math.exp(-1.0*abs(hand_initpos_err_left))),
            ('bonus', 1.*(dist_left<2*0) + 1.*(dist_left<0)),
            ('act_reg', -1.*act_mag),
            ("fin_open", np.exp(-20 * fin_open)),  # fin_open + np.log(fin_open +1e-8)
            ("pinky_close", np.exp(-25 * pinky_close)),  # fin_open + np.log(fin_open +1e-8)

            #('grip_bonus', 1.0 * grip_left),
            ('penalty', -1.*drop),
            ('sparse', dist_left < 0.055),
            #('sparse', 1.0 * grip_left - dist_left),
            ('solved', dist_left < 0.001),
            #('solved', grip_left and dist_left < 0.015),
            ('done', dist_left > 0.9),
        ))
        
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        
        return rwd_dict
    
    def reset(self, **kwargs):
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs


