""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import math
from myosuite.utils import gym
# import mujoco

from myosuite.envs.myo.base_v0 import BaseV0


class WheelHoldFixedEnvV0(BaseV0):

    # ⬇️ Keep exactly as you asked
    DEFAULT_OBS_KEYS = ['time', 'hand_qpos', 'hand_qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 2.0,
        "penalty": 5.0,
        "return_rwd": 50.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        # --- existing sites you used ---
        self.palm_r = self.sim.model.site_name2id("palm_r")
        self.hand_start_right = self.sim.model.site_name2id("hand_start_right")  # red, static

        # joint target (hand closing on rail) used by pose_err
        self.target_jnt_value = self.sim.model.key_qpos[0].copy()[13:]  # keyframe 0

        # ⬇️ NEW: grip-related references (names taken from your previous working env)
        # Fingertip sites for grip closure metric
        self.fin0 = self.sim.model.site_name2id("THtip")
        self.fin1 = self.sim.model.site_name2id("IFtip")
        self.fin2 = self.sim.model.site_name2id("MFtip")
        self.fin3 = self.sim.model.site_name2id("RFtip")
        self.fin4 = self.sim.model.site_name2id("LFtip")

        # Contact check: palm body vs handrail body
        self.palm_body_id = self.sim.model.body_name2id("thirdmc")
        self.rail_body_id = self.sim.model.body_name2id("right_handrail")

        # optional: used for gentle surface shaping (not required)
        self.rail_radius = 0.277

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs)
        # Keep your “returning” keyframe for reset
        self.init_qpos = self.sim.model.key_qpos[1].copy()

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[13:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[12:].copy()*self.dt

        # keep pose_err (drives the hand to close/back like your original)
        self.obs_dict['pose_err'] = self.target_jnt_value - self.obs_dict['hand_qpos']

        # ⬇️ extra (not in obs vector, but used for reward shaping)
        self.obs_dict["palm_pos"] = self.sim.data.site_xpos[self.palm_r].copy()
        self.obs_dict['fin0'] = self.sim.data.site_xpos[self.fin0].copy()
        self.obs_dict['fin1'] = self.sim.data.site_xpos[self.fin1].copy()
        self.obs_dict['fin2'] = self.sim.data.site_xpos[self.fin2].copy()
        self.obs_dict['fin3'] = self.sim.data.site_xpos[self.fin3].copy()
        self.obs_dict['fin4'] = self.sim.data.site_xpos[self.fin4].copy()

        # “return” vector for your original return_rwd
        self.obs_dict['return_err'] = self.sim.data.site_xpos[self.palm_r] - self.sim.data.site_xpos[self.hand_start_right]

        if self.sim.model.na > 0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        _, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[13:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[12:].copy()*self.dt

        obs_dict['pose_err'] = self.target_jnt_value - obs_dict['hand_qpos']
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_r].copy()
        obs_dict['return_err'] = sim.data.site_xpos[self.palm_r] - sim.data.site_xpos[self.hand_start_right]

        # fingertips for grip metric
        obs_dict['fin0'] = sim.data.site_xpos[self.fin0].copy()
        obs_dict['fin1'] = sim.data.site_xpos[self.fin1].copy()
        obs_dict['fin2'] = sim.data.site_xpos[self.fin2].copy()
        obs_dict['fin3'] = sim.data.site_xpos[self.fin3].copy()
        obs_dict['fin4'] = sim.data.site_xpos[self.fin4].copy()

        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # === original terms (unchanged) ===
        far_th = 4*np.pi / 2  # same as before
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)

        act_mag = 0.0
        if self.sim.model.na != 0:
            act_mag = np.linalg.norm(self.obs_dict.get('act', 0.0), axis=-1)
            act_mag = act_mag / self.sim.model.na

        pose_thd = 0.8
        return_err = np.linalg.norm(obs_dict["return_err"])
        return_rwd = math.exp(-2.0 * abs(return_err))

        # === NEW: grip shaping (added to dense; weights dict remains unchanged) ===
        # contact: palm touching rail
        palm_touching_rail = False
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            b1 = self.sim.model.geom_bodyid[con.geom1]
            b2 = self.sim.model.geom_bodyid[con.geom2]
            if ((b1 == self.palm_body_id and b2 == self.rail_body_id) or
                (b2 == self.palm_body_id and b1 == self.palm_body_id)):
                palm_touching_rail = True
                break
            # NOTE: the second condition has a typo (b1==palm). Fix to rail below:
            # Keeping loop simple; corrected check:
            if ((b1 == self.palm_body_id and b2 == self.rail_body_id) or
                (b2 == self.palm_body_id and b1 == self.rail_body_id)):
                palm_touching_rail = True
                break

        palm_touch_rwd = 1.0 if palm_touching_rail else 0.0  # binary

        # fingertip closure: encourage fingertips near palm
        palm_pos = obs_dict['palm_pos'].squeeze()
        fin_keys = ['fin0', 'fin1', 'fin2', 'fin3', 'fin4']
        fin_dists = [np.linalg.norm(obs_dict[k].squeeze() - palm_pos) for k in fin_keys]
        fin_open = float(sum(fin_dists)) / len(fin_dists)
        grip_close_rwd = math.exp(-8.0 * fin_open)

        # gentle surface shaping to avoid hovering too far from rail (optional)
        rail_center = self.sim.data.body_xpos[self.rail_body_id]
        palm_to_surface = max(0.0, np.linalg.norm(palm_pos - rail_center) - self.rail_radius)
        surface_shaping = math.exp(-10.0 * palm_to_surface)

        # === assemble rewards ===
        rwd_dict = collections.OrderedDict((
            # your original keys (kept)
            ('return_rwd', np.array([[return_rwd]])),
            ('pose', -pose_dist),
            ('bonus', 1.0 * (pose_dist < pose_thd) + 1.0 * (pose_dist < 1.5 * pose_thd)),
            ('penalty', -1.0 * ((pose_dist > far_th/2))),
            ('act_reg', -1.0 * act_mag),  # not in weights: regularizer only
            # new grip terms (named for logging; not in weights dict)
            ('palm_touch_rwd', palm_touch_rwd),
            ('grip_close_rwd', grip_close_rwd),
            ('surface_shaping', surface_shaping),
            # book-keeping
            ('sparse', -1.*pose_dist),
            ('solved', pose_dist < pose_thd),
            ('done', pose_dist > far_th),
        ))

        # Dense = (your weighted sum) + fixed-coeff grip shaping
        weighted = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        grip_combo = (8.0 * rwd_dict['palm_touch_rwd']   # strong push to make contact
                      + 5.0 * rwd_dict['grip_close_rwd'] # then close fingers
                      + 0.5 * rwd_dict['surface_shaping'])
        rwd_dict['dense'] = weighted + grip_combo + 0.1 * rwd_dict['act_reg']
        return rwd_dict

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.sim.data.qpos[:] = self.sim.model.key_qpos[1]
        self.sim.data.qvel[:] = 0
        if self.sim.model.na > 0:
            self.sim.data.act[:] = 0
        self.sim.forward()
        return obs
