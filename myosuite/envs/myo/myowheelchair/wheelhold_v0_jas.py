""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

# TRAINING RETURN

import collections
import numpy as np
import math
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['time', 'wheel_angle', 'hand_qpos', 'hand_qvel', 'return_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "return_rwd": 30.0,
        "hand_err_rwd": 10.0,

        "dist_reward": 5.0,
        "palm_touch_rwd": 5.0,
        "wheel_rotation": 25.0,
        "fin_open": -2.0,

        "bonus": 1.0, 
        "penalty": 1.0,
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
        # self.goal_sid_right = self.sim.model.site_name2id("wheelchair_grip_right") #green, movable
        self.palm_r = self.sim.model.site_name2id("palm_r")
        self.hand_start_right = self.sim.model.site_name2id("hand_start_right") #red, static
        # self.rail_bottom_right = self.sim.model.site_name2id("rail_bottom_right") #blue, movable
        
        # hand target position, start returning when reached
        self.hand_TARGET_right = self.sim.model.site_name2id("hand_TARGET_right") #blue, STATIC

        # define the palm and tip site id.
        # self.palm_r = self.sim.model.site_name2id('S_grasp')
        self.init_palm_z = self.sim.data.site_xpos[self.palm_r][-1]
        self.fin0 = self.sim.model.site_name2id("THtip")
        self.fin1 = self.sim.model.site_name2id("IFtip")
        self.fin2 = self.sim.model.site_name2id("MFtip")
        self.fin3 = self.sim.model.site_name2id("RFtip")
        self.fin4 = self.sim.model.site_name2id("LFtip")

        self.wheel_joint_id = self.sim.model.joint_name2id("right_rear") #right wheel joint
        self.init_wheel_angle = self.sim.data.qpos[self.wheel_joint_id].copy() #INIITAL wheel angle

        #elbow jnt
        self.elbow_joint_id = self.sim.model.joint_name2id("elbow_flexion")

        #phases check
        self.task_phase = "push"
        self.triggered_return_bonus = False
        self.triggered_push_bonus = False
        # self.prev_rwd_dict = None

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        
        self.init_qpos[25] = 0.64411


    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[13:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[12:].copy()*self.dt

        # self.obs_dict['wheel_err_right'] = self.sim.data.site_xpos[self.goal_sid_right] - self.sim.data.site_xpos[self.palm_r]
        # self.obs_dict['hand_initpos_err_right'] = self.sim.data.site_xpos[self.hand_start_right]- self.sim.data.site_xpos[self.goal_sid_right]

        self.obs_dict["palm_pos"] = self.sim.data.site_xpos[self.palm_r]
        self.obs_dict['fin0'] = self.sim.data.site_xpos[self.fin0]
        self.obs_dict['fin1'] = self.sim.data.site_xpos[self.fin1]
        self.obs_dict['fin2'] = self.sim.data.site_xpos[self.fin2]
        self.obs_dict['fin3'] = self.sim.data.site_xpos[self.fin3]
        self.obs_dict['fin4'] = self.sim.data.site_xpos[self.fin4]

        # self.obs_dict["rail_bottom_right"] = self.sim.data.site_xpos[self.rail_bottom_right]

        self.obs_dict['wheel_angle'] = np.array([self.sim.data.qpos[self.wheel_joint_id]])
        self.obs_dict["task_phase"] = np.array([1.0 if self.task_phase == "push" else -1.0])

        #calculate palm from target distance
        self.obs_dict['hand_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_TARGET_right]

        #calculate palm to return position
        self.obs_dict['return_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_start_right]

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[13:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[12:].copy()*self.dt

        # obs_dict['wheel_err_right'] = sim.data.site_xpos[self.goal_sid_right] - sim.data.site_xpos[self.palm_r]
        # obs_dict['hand_initpos_err_right'] = sim.data.site_xpos[self.hand_start_right]- sim.data.site_xpos[self.goal_sid_right]
        
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_r]
        obs_dict['fin0'] = sim.data.site_xpos[self.fin0]
        obs_dict['fin1'] = sim.data.site_xpos[self.fin1]
        obs_dict['fin2'] = sim.data.site_xpos[self.fin2]
        obs_dict['fin3'] = sim.data.site_xpos[self.fin3]
        obs_dict['fin4'] = sim.data.site_xpos[self.fin4]

        # obs_dict["rail_bottom_right"] = sim.data.site_xpos[self.rail_bottom_right]
        obs_dict['wheel_angle'] = np.array([sim.data.qpos[self.wheel_joint_id]])

        obs_dict["task_phase"] = np.array([1.0 if self.task_phase == "push" else -1.0])

        #calculate palm from target distance
        obs_dict['hand_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_TARGET_right]

        #calculate palm to return position
        obs_dict['return_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_start_right]

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # palm position from green dot (WARNING: MOVES WITH RAIL)
        # dist_right = np.linalg.norm(obs_dict['wheel_err_right']) 

        # CHECK IF PALM TOUCHING WHEEL
        palm_body_id = self.sim.model.body_name2id("thirdmc")
        rail_body_id = self.sim.model.body_name2id("right_handrail")
        palm_touching_rail = False
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            body1 = self.sim.model.geom_bodyid[con.geom1]
            body2 = self.sim.model.geom_bodyid[con.geom2]
            if ((body1 == palm_body_id and body2 == rail_body_id) or
                (body2 == palm_body_id and body1 == rail_body_id)):
                palm_touching_rail = True
                break
        ### SPARSE reward for palm touching rail ###
        palm_touch_rwd = 5.0 if palm_touching_rail else 0.0

        ### DENSE reward for palm close to rail ###
        rail_center_pos = self.sim.data.body_xpos[rail_body_id]
        # Distance from palm to the rail *surface*
        palm_to_rail_surface_dist = max(0.0, np.linalg.norm(obs_dict["palm_pos"] - rail_center_pos) - 0.277)
        # Reward: higher when palm is closer to the surface
        dist_reward = np.exp(-10.0 * palm_to_rail_surface_dist)

        # calculate wheel rotation, just want it as big and negative as possible
        wheel_angle_now = self.sim.data.qpos[self.wheel_joint_id]
        wheel_rotation = -1*(wheel_angle_now - self.init_wheel_angle) #rotate cw? I think?

        # minimize muscle activation for realism
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        # gripping, minimize distance between palm and fingertips
        fin_keys = ['fin0', 'fin1', 'fin2', 'fin3', 'fin4']
        fin_open = sum(
            np.linalg.norm(obs_dict[fin].squeeze() - obs_dict['palm_pos'].squeeze(), axis=-1)
            for fin in fin_keys
        )

        # ELBOW JOINT VALUE
        elbow_now = self.sim.data.qpos[self.elbow_joint_id]

        #errs
        hand_err = np.linalg.norm(obs_dict["hand_err"])
        return_err = np.linalg.norm(obs_dict["return_err"])

        drop = return_err>10.0

        rwd_dict = collections.OrderedDict((
            ('return_rwd', 5.0 * np.exp(-10.0 * return_err)),
            ('hand_err_rwd', 0),
            ('dist_reward', 0),
            ('palm_touch_rwd', 0.5*palm_touch_rwd),
            ('wheel_rotation', 0),
            ('act_reg', -1. * act_mag),
            ('fin_open', 0),
            ('bonus', 1.*(return_err<2*0.1) + 1.*(return_err<0.1)),
            ('penalty', -1.*(drop)),
            ('sparse', -10.0*return_err),
            # ('solved', 0),
            # ('done', 0),
            ('solved', (return_err < 0.0025) and (not drop)),
            ('done', drop),
        ))
        
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        # self.prev_rwd_dict = rwd_dict  # Always update
        return rwd_dict
    
    def reset(self, **kwargs):
        self.robot.sync_sims(self.sim, self.sim_obsd)
        # Get default initial positions
        qpos = self.sim.data.qpos.copy()
        # Randomize hand joint positions (assuming indices 13: are hand joints)
        qpos[13:] += self.np_random.uniform(
            low=0, high=0.5, size=(len(qpos)-13)
        )
        qpos[25] = 0.64411 # move shoulder joint up
        obs = super().reset(reset_qpos=qpos, **kwargs)
        return obs