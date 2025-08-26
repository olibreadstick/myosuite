""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import math
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0Left(BaseV0):

    DEFAULT_OBS_KEYS = ['time', 'wheel_angle', 'hand_qpos', 'hand_qvel', 'wheel_angle_l', 'hand_qpos_l', 'hand_qvel_l', 'pose_err', 'pose_err_l']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
        "return_rwd": 5.0,
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
        self.palm_l = self.sim.model.site_name2id("palm_l")
        self.hand_start_left = self.sim.model.site_name2id("hand_start_left") #red, static
        # hand target position, start returning when reached
        self.hand_TARGET_left = self.sim.model.site_name2id("hand_TARGET_left") #blue, STATIC

        self.palm_r = self.sim.model.site_name2id("palm_r")
        self.hand_start_right = self.sim.model.site_name2id("hand_start_right") #red, static        
        # hand target position, start returning when reached
        self.hand_TARGET_right = self.sim.model.site_name2id("hand_TARGET_right") #blue, STATIC

        # define the palm and tip site id.
        # self.palm_r = self.sim.model.site_name2id('S_grasp')
        self.init_palm_z_l = self.sim.data.site_xpos[self.palm_l][-1]
        self.init_palm_z = self.sim.data.site_xpos[self.palm_r][-1]

        self.wheel_joint_id_l = self.sim.model.joint_name2id("left_rear") #right wheel joint
        self.init_wheel_angle_l = self.sim.data.qpos[self.wheel_joint_id_l].copy() #INIITAL wheel angle

        self.wheel_joint_id = self.sim.model.joint_name2id("right_rear") #right wheel joint
        self.init_wheel_angle = self.sim.data.qpos[self.wheel_joint_id].copy() #INIITAL wheel angle

        self.target_jnt_value = self.sim.model.key_qpos[0].copy()

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        
        self.init_qpos = self.sim.model.key_qpos[1].copy() # copy returning keyframe


    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[13:49].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[12:48].copy()*self.dt

        self.obs_dict['hand_qpos_l'] = self.sim.data.qpos[50:].copy()
        self.obs_dict['hand_qvel_l'] = self.sim.data.qvel[49:].copy()*self.dt

        self.obs_dict["palm_pos_l"] = self.sim.data.site_xpos[self.palm_l]
        self.obs_dict["palm_pos_r"] = self.sim.data.site_xpos[self.palm_r]

        self.obs_dict['wheel_angle'] = np.array([self.sim.data.qpos[self.wheel_joint_id]])
        self.obs_dict['wheel_angle_l'] = np.array([self.sim.data.qpos[self.wheel_joint_id_l]])

        self.obs_dict['pose_err'] = self.target_jnt_value[13:49] - self.obs_dict['hand_qpos']
        self.obs_dict['pose_err_l'] = self.target_jnt_value[50:] - self.obs_dict['hand_qpos_l']

        #calculate palm to return position
        self.obs_dict['return_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_start_right]
        self.obs_dict['return_err_l'] = self.sim.data.site_xpos[self.palm_l]- self.sim.data.site_xpos[self.hand_start_left]

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[13:49].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[12:48].copy()*self.dt

        obs_dict['hand_qpos_l'] = sim.data.qpos[50:].copy()
        obs_dict['hand_qvel_l'] = sim.data.qvel[49:].copy()*self.dt

        obs_dict["palm_pos_l"] = sim.data.site_xpos[self.palm_l]
        obs_dict["palm_pos_r"] = sim.data.site_xpos[self.palm_r]

        obs_dict['wheel_angle'] = np.array([sim.data.qpos[self.wheel_joint_id]])
        obs_dict['wheel_angle_l'] = np.array([sim.data.qpos[self.wheel_joint_id_l]])

        obs_dict['pose_err'] = self.target_jnt_value[13:49] - obs_dict['hand_qpos']
        obs_dict['pose_err_l'] = self.target_jnt_value[50:] - obs_dict['hand_qpos_l']

         #calculate palm to return position
        obs_dict['return_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_start_right]
        obs_dict['return_err_l'] = self.sim.data.site_xpos[self.palm_l]- self.sim.data.site_xpos[self.hand_start_left]


        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        far_th = 4 * np.pi / 2

        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        pose_dist_l = np.linalg.norm(obs_dict['pose_err_l'], axis=-1)

        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)
        if self.sim.model.na != 0:
            act_mag = act_mag / self.sim.model.na

        pose_thd = 0.6

        # Combine left and right by summing (or you could use np.mean([...]) to average)
        pose_total = pose_dist + pose_dist_l
        bonus_total = (
            1.0 * (pose_dist < pose_thd) + 1.0 * (pose_dist < 1.5 * pose_thd) +
            1.0 * (pose_dist_l < pose_thd) + 1.0 * (pose_dist_l < 1.5 * pose_thd)
        )
        penalty_total = -1.0 * ((pose_dist > far_th) + (pose_dist_l > far_th))
        sparse_total = -1.0 * (pose_dist + pose_dist_l)
        solved_total = (pose_dist < pose_thd) or (pose_dist_l < pose_thd)
        done_total = (pose_dist > far_th) or (pose_dist_l > far_th)

        return_err = np.linalg.norm(obs_dict["return_err"])
        return_err_l = np.linalg.norm(obs_dict["return_err_l"])
        return_rwd= math.exp(-5.0 * abs(return_err)) + math.exp(-5.0 * abs(return_err_l))

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('return_rwd', np.array([[return_rwd]])),
            ('pose', -1.0 * pose_total),
            ('bonus', bonus_total),
            ('penalty', penalty_total),
            ('act_reg', -1.0 * act_mag),
            # Must keys
            ('sparse', sparse_total),
            ('solved', solved_total),
            ('done', done_total),
        ))

        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        return rwd_dict
    
    def reset(self, **kwargs):
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)

        return obs