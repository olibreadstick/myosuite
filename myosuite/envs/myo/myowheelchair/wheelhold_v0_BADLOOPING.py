""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import math
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['time', 'wheel_err_right', 'wheel_angle', 'hand_qpos', 'hand_qvel', 'task_phase']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal_dist": 15.0,
        "hand_dist" : 5.0,
        "fin_open": 15.0,
        "bonus": 0.0,
        "penalty": 2,
        "wheel_rotation": 0.0,
        "rotation_bonus": 0.0
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
            weighted_reward_keys: dict = None,  # unused now
            **kwargs,
        ):
        self.goal_sid_right = self.sim.model.site_name2id("wheelchair_grip_right")
        self.palm_r = self.sim.model.site_name2id("palm_r")
        self.hand_start_right = self.sim.model.site_name2id("hand_start_right")
        self.rail_bottom_right = self.sim.model.site_name2id("rail_bottom_right")

        # define the palm and tip site id.
        # self.palm_r = self.sim.model.site_name2id('S_grasp')
        self.init_palm_z = self.sim.data.site_xpos[self.palm_r][-1]
        self.fin0 = self.sim.model.site_name2id("THtip")
        self.fin1 = self.sim.model.site_name2id("IFtip")
        self.fin2 = self.sim.model.site_name2id("MFtip")
        self.fin3 = self.sim.model.site_name2id("RFtip")
        self.fin4 = self.sim.model.site_name2id("LFtip")

        self.wheel_joint_id = self.sim.model.joint_name2id("right_rear")
        self.init_wheel_angle = self.sim.data.qpos[self.wheel_joint_id]

        self.task_phase = "push"
        self.push_threshold = np.deg2rad(30)  # ~0.52 radians
        self.return_threshold = 0.03  # distance in meters

        self.initial_hand_pos = self.sim.data.site_xpos[self.palm_r].copy()
        self.initial_wheel_angle = self.sim.data.qpos[self.wheel_joint_id].copy()

        self.rwd_weights_push = {
            "goal_dist": 15.0,
            "hand_dist": 5.0,
            "fin_open": 15.0,
            "penalty": 2.0,
            "wheel_rotation": 5.0,
            "rotation_bonus": 2.0,
        }

        self.rwd_weights_return = {
            "hand_dist": 20.0,        # reward for going back to initial hand pos
            "fin_open": 5.0,
            "penalty": 2.0,
        }

        #self.goal_sid_left = self.sim.model.site_name2id("wheel_grip_goal_left")
        #self.object_init_pos = self.sim.data.site_xpos[self.object_sid].copy()

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys={},  # <- pass dummy dict
            **kwargs
        )        
        self.init_qpos = self.sim.model.key_qpos[0].copy() # copy the sitting + grabbing wheels keyframe


    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[13:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[12:].copy()*self.dt
        #self.obs_dict['wheel_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['wheel_err_right'] = self.sim.data.site_xpos[self.goal_sid_right] - self.sim.data.site_xpos[self.palm_r]
        self.obs_dict['hand_initpos_err_right'] = self.sim.data.site_xpos[self.hand_start_right]- self.sim.data.site_xpos[self.goal_sid_right]

        self.obs_dict["palm_pos"] = self.sim.data.site_xpos[self.palm_r]
        self.obs_dict['fin0'] = self.sim.data.site_xpos[self.fin0]
        self.obs_dict['fin1'] = self.sim.data.site_xpos[self.fin1]
        self.obs_dict['fin2'] = self.sim.data.site_xpos[self.fin2]
        self.obs_dict['fin3'] = self.sim.data.site_xpos[self.fin3]
        self.obs_dict['fin4'] = self.sim.data.site_xpos[self.fin4]

        self.obs_dict["rail_bottom_right"] = self.sim.data.site_xpos[self.rail_bottom_right]

        self.obs_dict['wheel_angle'] = np.array([self.sim.data.qpos[self.wheel_joint_id]])

        self.obs_dict["task_phase"] = np.array([1.0 if self.task_phase == "push" else -1.0])

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
        obs_dict['hand_initpos_err_right'] = sim.data.site_xpos[self.hand_start_right]- sim.data.site_xpos[self.goal_sid_right]
        #add the initial and end target points
        #could add the fingertips here,
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_r]
        obs_dict['fin0'] = sim.data.site_xpos[self.fin0]
        obs_dict['fin1'] = sim.data.site_xpos[self.fin1]
        obs_dict['fin2'] = sim.data.site_xpos[self.fin2]
        obs_dict['fin3'] = sim.data.site_xpos[self.fin3]
        obs_dict['fin4'] = sim.data.site_xpos[self.fin4]

        obs_dict["rail_bottom_right"] = sim.data.site_xpos[self.rail_bottom_right]

        obs_dict['wheel_angle'] = np.array([sim.data.qpos[self.wheel_joint_id]])

        obs_dict["task_phase"] = np.array([1.0 if self.task_phase == "push" else -1.0])

        #obs_dict['wheel_err_left'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # --- Positions and angles ---
        dist_right = np.linalg.norm(obs_dict['wheel_err_right'])
        palm_pos = obs_dict['palm_pos']
        hand_return_dist = np.linalg.norm(palm_pos - self.initial_hand_pos)

        current_wheel_angle = self.sim.data.qpos[self.wheel_joint_id]
        wheel_rotation = current_wheel_angle - self.initial_wheel_angle
        wheel_rotation_target = np.pi / 2
        wheel_rotation_err = abs(wheel_rotation - wheel_rotation_target)
        wheel_rotation_rwd = math.exp(-5.0 * wheel_rotation_err)

        drop = dist_right > 0.5
        act_mag = np.linalg.norm(self.obs_dict['act']) / self.sim.model.na if self.sim.model.na != 0 else 0

        # --- Finger openness ---
        fin_keys = ['fin0', 'fin1', 'fin2', 'fin3', 'fin4']
        fin_open = sum(
            np.linalg.norm(obs_dict[fin] - obs_dict['rail_bottom_right'])
            for fin in fin_keys
        )

        # --- Phase switching ---
        if self.task_phase == "push" and wheel_rotation > self.push_threshold:
            self.task_phase = "return"
        elif self.task_phase == "return" and hand_return_dist < self.return_threshold:
            self.task_phase = "push"
            self.initial_wheel_angle = current_wheel_angle

        # --- Rewards shared across phases ---
        goal_dist_rwd = math.exp(-2.0 * dist_right)
        hand_dist_rwd = math.exp(-1.0 * hand_return_dist)
        fin_open_rwd = math.exp(-20.0 * fin_open)
        act_reg = -1.0 * act_mag
        penalty = -1.0 * drop
        bonus = 1.0 * (dist_right < 0.1) + 1.0 * (dist_right < 0.05)  # can tweak
        solved = dist_right < 0.001 and wheel_rotation_err < 0.05
        done = dist_right > 0.9

        # --- Sparse reward ---
        sparse = float(dist_right < 0.055) if self.task_phase == "push" else float(hand_return_dist < self.return_threshold)

        # --- Assemble reward dict ---
        rwd_dict = collections.OrderedDict()
        rwd_dict["goal_dist"] = goal_dist_rwd
        rwd_dict["hand_dist"] = hand_dist_rwd
        rwd_dict["fin_open"] = fin_open_rwd
        rwd_dict["bonus"] = bonus
        rwd_dict["act_reg"] = act_reg
        rwd_dict["penalty"] = penalty
        rwd_dict["wheel_rotation"] = wheel_rotation_rwd
        rwd_dict["rotation_bonus"] = 1.0 if wheel_rotation_err < 0.05 else 0.0
        rwd_dict["sparse"] = sparse
        rwd_dict["solved"] = solved
        rwd_dict["done"] = done

        # --- Use phase-specific weights ---
        weights = self.rwd_weights_push if self.task_phase == "push" else self.rwd_weights_return
        rwd_dict["dense"] = sum(weights.get(k, 0.0) * rwd_dict[k] for k in rwd_dict)

        # --- Debug info ---
        rwd_dict["phase"] = 1.0 if self.task_phase == "push" else -1.0
        rwd_dict["wheel_angle"] = current_wheel_angle
        rwd_dict["hand_return_dist"] = hand_return_dist

        return rwd_dict


    def reset(self, **kwargs):
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs
