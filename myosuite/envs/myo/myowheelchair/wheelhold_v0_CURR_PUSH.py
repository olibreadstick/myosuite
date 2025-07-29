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

    DEFAULT_OBS_KEYS = ['time', 'wheel_angle', 'hand_qpos', 'hand_qvel']
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

        self.triggered_return_bonus = False
        self.init_wheel_angle = None


        self._setup(**kwargs)


    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.time_step = 0
        self.return_phase_start = 40
        
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
        #self.init_wheel_angle = self.sim.data.qpos[self.wheel_joint_id].copy() #INIITAL wheel angle

        #elbow jnt
        self.elbow_joint_id = self.sim.model.joint_name2id("elbow_flexion")

        #phases check
        self.task_phase = "push"
        # self.prev_rwd_dict = None

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        
        self.init_qpos = self.sim.model.key_qpos[0].copy() # copy the sitting + grabbing wheels keyframe




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
        # self.obs_dict["task_phase"] = np.array([1.0 if self.task_phase == "push" else -1.0])

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

        # obs_dict["task_phase"] = np.array([1.0 if self.task_phase == "push" else -1.0])

        #calculate palm from target distance
        obs_dict['hand_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_TARGET_right]

        #calculate palm to return position
        obs_dict['return_err'] = self.sim.data.site_xpos[self.palm_r]- self.sim.data.site_xpos[self.hand_start_right]

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        return_phase = self.time_step >= self.return_phase_start
        
        
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
        
        if self.init_wheel_angle is None:
             self.init_wheel_angle = wheel_angle_now.copy()
             print(f"[Fallback] init_wheel_angle was unset, now set to {self.init_wheel_angle:.3f}")

        
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

        # if return_phase:
        # # Return phase: only use return_reward
        #     return_reward = np.exp(-50.0 * return_err) - 0.5 * return_err
        #     rwd_dict = collections.OrderedDict((
        #         ('return_rwd', return_reward),
        #         ('hand_err_rwd', 0),
        #         ('dist_reward', 0.5*dist_reward),
        #         ('palm_touch_rwd', 0),
        #         ('wheel_rotation', 0),
        #         ('act_reg', -0.25 * act_mag),
        #         ('fin_open', np.exp(fin_open)),
        #         ('bonus', 0),
        #         ('penalty', 0),
        #         ('sparse', return_err < 0.025),
        #         ('solved', 0),
        #         ('done', 0),
        #     ))
        #         # ('solved', return_err < 0.0025),
        #         # ('done', return_err > 50.0),
        #     print(f"[Return] Step: {self.time_step}, err: {return_err:.3f}, reward: {return_reward:.3f}")
        # else:
        #     # Normal phase: compute all rewards
        #     rwd_dict = collections.OrderedDict((
        #         ('return_rwd', 0),
        #         ('hand_err_rwd', math.exp(-20.0 * abs(hand_err))),
        #         ('dist_reward', dist_reward),
        #         ('palm_touch_rwd', palm_touch_rwd),
        #         ('wheel_rotation', 15.0 * wheel_rotation),
        #         ('act_reg', -0.5 * act_mag),
        #         ('fin_open', np.exp(-5.0 * fin_open)),
        #         ('bonus', 1.0 * (wheel_angle_now < self.init_wheel_angle)),
        #         ('penalty', -1.0 * (wheel_angle_now > self.init_wheel_angle)),
        #         ('sparse', 0),
        #         ('solved', 0),
        #         ('done', 0)
        #     ))

    
        if self.task_phase == "push":
            # print(f"[Push Phase] hand_err: {hand_err:.4f}, elbow: {elbow_now:.4f}, wheel: {wheel_rotation:.4f}")

<<<<<<< Updated upstream:myosuite/envs/myo/myowheelchair/wheelhold_v0_CURR_PUSH.py
            triggered_return =  hand_err < 0.05 and wheel_rotation < -0.0012
            bonus_reward = 200.0 if (triggered_return and not self.triggered_return_bonus) else 0.0
=======
            #triggered_return =  hand_err < 0.05 and wheel_rotation < -0.0012
            triggered_return = (abs(wheel_rotation) >= np.deg2rad(0.01))
            bonus_reward = 300.0 if (triggered_return and not self.triggered_return_bonus) else 0.0
>>>>>>> Stashed changes:myosuite/envs/myo/myowheelchair/wheelhold_v0.py

            rwd_dict = collections.OrderedDict((
                ('return_rwd', 0),
                ('hand_err_rwd', math.exp(-20.0 * abs(hand_err))),
                ('dist_reward', dist_reward),
                ('palm_touch_rwd', palm_touch_rwd),
                ('wheel_rotation', 15.0 * wheel_rotation),
                ('act_reg', -0.5 * act_mag),
                ('fin_open', np.exp(-5.0 * fin_open)),
                ('bonus', 5.0 * (wheel_angle_now < self.init_wheel_angle) + bonus_reward),
                ('penalty', -1.0 * (wheel_angle_now > self.init_wheel_angle)),
                ('sparse', 0),
                ('solved', 0),
                ('done', 0),
                # ('solved', wheel_rotation < -500.0),
                # ('done', wheel_rotation > 500.0),
            ))
            
            # print(f"[Push] Step: init = {self.init_wheel_angle:.3f}, now = {wheel_angle_now:.3f}, rotation = {wheel_rotation:.3f}")
            # if abs(wheel_rotation) >= (np.pi / 18):
            #     print("10 deg reached")
            #     print("returning")
            #     self.task_phase = "return"
            
            print(f"[Push] Step: init = {self.init_wheel_angle:.6f}, now = {wheel_angle_now:.6f}, rotation = {wheel_rotation:.6f}")
            if triggered_return and not self.triggered_return_bonus:
                print("returning")
                self.task_phase = "return"
                self.triggered_return_bonus = True  # Prevent repeating the bonus
                self.triggered_push_bonus = False

        elif self.task_phase == "return":
            # print(f"[Return] err: {return_err:.3f}, elbow: {elbow_now:.4f}, wheel: {wheel_rotation:.4f}")

            triggered_push = return_err < 0.05
            bonus_reward = 400.0 if triggered_push and not self.triggered_push_bonus else 0.0

            rwd_dict = collections.OrderedDict((
                ('return_rwd', 5.0*math.exp(-50.0 * abs(return_err))),
                ('hand_err_rwd', 0),
                ('dist_reward', 0),
                ('palm_touch_rwd', 0),
                ('wheel_rotation', 0),
                ('act_reg', -0.25 * act_mag),
                ('fin_open', 0),
                ('bonus', 1.* (return_err < 0.2)+ bonus_reward),
                ('penalty', -return_err),
                ('sparse', return_err < 0.1),
                ('solved', 0),
                ('done', 0),
                # ('solved', return_err < 0.0025),
                # ('done', return_err > 50.0),
            ))
            #if return_err < 0.05 and elbow_now < -1.0:
            if elbow_now < -.4:
                print("return successful, pushing again")
                self.task_phase = "push"
                self.triggered_push_bonus = True  # prevent repeat bonus
                self.triggered_return_bonus = False  # reset for next cycle
        else:
            raise ValueError(f"Unknown task phase: {self.task_phase}")

        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        # self.prev_rwd_dict = rwd_dict  # Always update
        return rwd_dict
    
    def reset(self, **kwargs):
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        self.time_step = 0
        # self.prev_rwd_dict = self.get_reward_dict(self.get_obs_dict(self.sim))  # optional init

        self.init_wheel_angle = self.sim.data.qpos[self.wheel_joint_id].copy()

        return obs

    def step(self, action):
        self.time_step += 1
        return super().step(action)