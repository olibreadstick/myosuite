# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""High-level boxing: pick the next skill clip against a scripted opponent.

Prototype of the high level of hierarchical boxing. The low level is a perfect
tracker: a chosen skill clip is played back kinematically (poses written to
``qpos``, no muscle physics), so the skill menu, observation and reward can be
designed before a physical tracking policy exists. Falls cannot happen here.

Each ``step`` is one decision: agent ``a0`` plays the chosen skill to its end
while the opponent (``a1``) runs a scripted policy. Every skill starts facing
the opponent and continues from where the previous one ended; a skill's travel
is shortened so the boxers keep ``min_separation_m`` and stay inside the ring.

Usage::

    env = SkillSelectEnv()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    build_boxing_vs_model,
)
from myosuite.envs.myo.tasks.challenge.boxing_vs.skill_library import (
    KINDS,
    NQ,
    SITE_KEYS,
    SkillClip,
    load_skill_clips,
    place_points,
    place_qpos,
)
from myosuite.terms.multiplayer.boxing_skill_reward import (
    fist_hit_damage,
    in_range,
    interrupt_mask,
)

AGENTS = ("agent_0", "agent_1")
PREFIX = ("a0", "a1")
FISTS = slice(0, 2)
HEAD, BODY, PELVIS = (
    SITE_KEYS.index(k) for k in ("head_zone", "body_zone", "pelvis_site")
)
PUNCH_KINDS = ("punch_R", "punch_L", "combo")
OPPONENTS = ("guard", "random", "pressure")


@dataclass
class SkillSelectConfig:
    """Skill menu, scripted opponents, hit model and reward weights.

    Distances are horizontal pelvis-to-pelvis separations unless noted.
    """

    menu: tuple[str, ...] = (
        "guard_short",
        "punch_R_push1",
        "punch_L_push1",
        "one_two_push",
        "punch_R_stand1",
        "punch_L_stand1",
        "step_fwd",
        "step_back",
    )
    opponent_menu: tuple[str, ...] | None = None  # None: same as ``menu``
    opponents: tuple[str, ...] = OPPONENTS  # scripted opponents, one drawn per episode
    self_play_prob: float = (
        0.8  # chance of a pool policy instead (once the pool has one)
    )
    pool_size: int = 10  # policy snapshots kept; the oldest is dropped
    round_s: float = 15.0
    separation_m: tuple[float, float] = (1.2, 2.0)  # start separation range
    min_separation_m: float = 0.35
    approach_stop_m: float = 0.45  # forward travel stops this far short of the opponent
    ring_half_m: float = 3.0  # roots stay within |x|, |y| <= this
    range_m: tuple[float, float] = (0.4, 0.65)  # punching distance
    whiff_distance_m: float = 0.85  # punches started beyond this are wasted

    head_radius_m: float = 0.15
    body_radius_m: float = 0.18
    w_head: float = 2.0
    w_body: float = 1.0
    hit_min_speed_mps: float = 1.5
    hit_full_speed_mps: float = 4.5
    block_factor: float = 0.3  # damage scale on a boxer in guard facing the attacker
    block_facing_cos: float = 0.7
    interrupt_factor: float = 0.0  # damage scale of a punch after its thrower is hit
    ko_damage: float = 10.0

    r_hit: float = 1.0  # per unit of damage dealt
    r_hurt: float = 1.5  # per unit of damage received (subtracted)
    r_block: float = 0.5  # per unit of damage a block prevented
    r_range: float = 0.05  # per second at punching distance
    r_whiff: float = -0.2  # per punch started out of range
    r_ko: float = 5.0  # KO won (+) or lost (-)

    record_frames: bool = False  # put the played ``qpos`` frames in ``info``
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "opp_rel",
            "opp_heading",
            "opp_kind",
            "opp_phase",
            "own_kind",
            "damage",
            "time_left",
            "ring_pos",
        ]
    )


@dataclass
class _Track:
    """A placed skill clip on the global frame clock."""

    clip: SkillClip
    t0: int
    qpos: np.ndarray
    sites: np.ndarray
    forward: np.ndarray  # horizontal facing direction
    interrupted: bool = False  # punch already cut short by a hit

    @property
    def end(self) -> int:
        return self.t0 + self.clip.n

    def frames(self, t0: int, t1: int) -> np.ndarray:
        """Clip frame indices of global frames ``[t0, t1)`` (holds the last frame)."""
        return np.clip(np.arange(t0, t1) - self.t0, 0, self.clip.n - 1)


class SkillSelectEnv(MyoGymnasiumEnv):
    """Choose ``a0``'s next skill clip; the opponent ``a1`` is scripted or a pool policy.

    Action: ``Discrete(len(menu))`` index of the next skill.

    Observation (``a0``'s heading frame, float32):
        ``opp_rel`` (3) opponent pelvis x, y and distance; ``opp_heading`` (2)
        cos/sin of the opponent's facing; ``opp_kind`` (7) one-hot of the
        opponent's current skill kind; ``opp_phase`` (2) its progress and frames
        left (/100); ``own_kind`` (7) own current skill kind; ``damage`` (2) damage
        received and dealt (/``ko_damage``); ``time_left`` (1) round fraction;
        ``ring_pos`` (2) own root x, y (/``ring_half_m``).

    A boxer hit while punching lands only ``interrupt_factor`` of the rest of
    that punch's damage.

    Reward per decision, summed over its frames: ``r_hit`` x damage dealt,
    ``-r_hurt`` x damage received, ``r_block`` x damage a block prevented (a hit
    landing on ``a0`` in guard, facing the attacker, does ``block_factor`` of its
    damage), ``r_range`` per second at punching distance,
    ``r_whiff`` for a punch started beyond ``whiff_distance_m`` and ``+-r_ko``.

    Args:
        config: Task configuration.
        render_mode: Gymnasium render mode.
    """

    def __init__(
        self, config: SkillSelectConfig | None = None, render_mode: str | None = None
    ) -> None:
        super().__init__(frame_skip=1, render_mode=render_mode)
        self.cfg = config or SkillSelectConfig()
        self.model, self.data, self.meta = build_boxing_vs_model(BoxingVsConfig())
        self._qpos_adr = [int(self.model.joint(f"{p}_root").qposadr[0]) for p in PREFIX]
        self.menu = list(self.cfg.menu)
        self.opp_menu = list(self.cfg.opponent_menu or self.cfg.menu)
        names = sorted({*self.menu, *self.opp_menu, "guard"})
        self.clips = load_skill_clips(
            self.model,
            self.meta.site_ids[AGENTS[0]],
            self._qpos_adr[0],
            self.meta.pelvis_body_id[AGENTS[0]],
            names,
        )
        self.guard_skill = next(
            (n for n in self.opp_menu if self.clips[n].kind == "guard"), "guard"
        )
        self.hz = self.clips["guard"].hz
        self._ctrl_dt = 1.0 / self.hz
        self.round_frames = round(self.cfg.round_s * self.hz)
        self.obs_keys = list(self.cfg.obs_keys)
        self.action_space = gym.spaces.Discrete(len(self.menu))
        self._pool: list[Callable[[np.ndarray], np.ndarray]] = []
        self._reset_state()
        self.observation_space = self._unbounded_obs_space(
            self._obs_dict_to_vec(self._get_obs_dict(None)).size
        )

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Clear every per-episode quantity (both boxers in guard, 1.6 m apart)."""
        self._t = 0
        self._damage = np.zeros(2)  # damage received by a0, a1
        self._blocked = 0.0  # damage a0's blocks prevented
        self._in_zone = np.zeros((2, 2), dtype=bool)
        self._opponent = OPPONENTS[0]
        self._opp_policy: Callable[[np.ndarray], np.ndarray] | None = None
        self._last_rwd: dict[str, Any] = {"dense": 0.0, "done": False}
        guard = self.clips["guard"]
        spawn = [np.array([0.0, 0.8]), np.array([0.0, -0.8])]
        self._tracks = [
            self._place(guard, spawn[i], spawn[1 - i], -guard.n) for i in range(2)
        ]

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Draw the start separation and angle and the opponent; place both boxers."""
        self._reset_state()
        lo, hi = self.cfg.separation_m
        sep = float(np_random.uniform(lo, hi))
        angle = float(np_random.uniform(-0.3, 0.3))  # tilt of the line between them
        half = 0.5 * sep * np.array([np.sin(angle), np.cos(angle)])
        spawn = [half, -half]
        guard = self.clips["guard"]
        self._tracks = [
            self._place(guard, spawn[i], spawn[1 - i], -guard.n) for i in range(2)
        ]
        if self._pool and np_random.random() < self.cfg.self_play_prob:
            self._opponent = "self"
            self._opp_policy = self._pool[int(np_random.integers(len(self._pool)))]
        else:
            self._opponent = str(np_random.choice(self.cfg.opponents))
        self._write_pose(-1)
        return {"opponent": self._opponent, "separation": sep}

    # ------------------------------------------------------------------
    # Skill placement
    # ------------------------------------------------------------------

    def _place(
        self, clip: SkillClip, xy: np.ndarray, toward: np.ndarray, t0: int
    ) -> _Track:
        """Place ``clip`` with its first root at ``xy``, facing the point ``toward``."""
        to = toward - xy
        dyaw = float(np.arctan2(to[1], to[0]) - np.arctan2(*clip.forward[::-1]))
        c, s = np.cos(dyaw), np.sin(dyaw)
        forward = np.array(
            [
                c * clip.forward[0] - s * clip.forward[1],
                s * clip.forward[0] + c * clip.forward[1],
            ]
        )
        return _Track(
            clip=clip,
            t0=t0,
            qpos=place_qpos(clip.qpos, xy, dyaw),
            sites=place_points(clip.sites, clip.qpos[0, :3], xy, dyaw),
            forward=forward,
        )

    def _sites(self, agent: int, t0: int, t1: int) -> np.ndarray:
        """``(t1 - t0, 5, 3)`` boxing-site positions of ``agent`` over ``[t0, t1)``."""
        tr = self._tracks[agent]
        return tr.sites[tr.frames(t0, t1)]

    def _schedule(self, agent: int, name: str, t: int) -> None:
        """Start skill ``name`` for ``agent`` at frame ``t`` from its current pose.

        The skill faces the opponent's pelvis. Its horizontal travel is scaled
        down (in 5 % steps) until the boxers keep ``min_separation_m``, the
        boxer gets no nearer than ``approach_stop_m`` along its facing direction
        (so a step never walks past the opponent) and the root stays in the ring.
        """
        prev = self._tracks[agent]
        clip = self.clips[name]
        own_xy = prev.qpos[-1, :2]
        opp_now = self._sites(1 - agent, t, t + 1)[0, PELVIS, :2]
        tr = self._place(clip, own_xy, opp_now, t)
        opp = self._sites(1 - agent, t, t + clip.n)[:, PELVIS, :2]
        travel = tr.qpos[:, :2] - tr.qpos[0, :2]
        ahead_now = float((opp[0] - tr.sites[0, PELVIS, :2]) @ tr.forward)
        stop = min(self.cfg.approach_stop_m, ahead_now)  # never forces a retreat
        for alpha in np.linspace(1.0, 0.0, 21):
            shift = (alpha - 1.0) * travel
            pelvis = tr.sites[:, PELVIS, :2] + shift
            root = tr.qpos[:, :2] + shift
            if (
                np.linalg.norm(pelvis - opp, axis=1).min() >= self.cfg.min_separation_m
                and ((opp - pelvis) @ tr.forward).min() >= stop - 1e-9
                and np.abs(root).max() <= self.cfg.ring_half_m
            ):
                break
        tr.qpos[:, :2] += shift
        tr.sites[..., :2] += shift[:, None]
        self._tracks[agent] = tr

    def _opponent_skill(self, t: int) -> str:
        """Next skill of the opponent (scripted, or drawn from a pool policy)."""
        rng = self.np_random
        if self._opponent == "self":
            probs = np.asarray(self._opp_policy(self._obs_vec(1, t)), dtype=np.float64)
            return self.menu[int(rng.choice(len(probs), p=probs / probs.sum()))]
        if self._opponent == "guard":
            return "guard"
        if self._opponent == "random":
            return str(rng.choice(self.opp_menu))
        dist = self._distance(t)
        punches = [n for n in self.opp_menu if self.clips[n].kind in PUNCH_KINDS]
        if dist > self.cfg.range_m[1] and "step_fwd" in self.opp_menu:
            return "step_fwd"
        if punches and rng.random() < 0.7:
            return str(rng.choice(punches))
        return self.guard_skill

    def _distance(self, t: int) -> float:
        p = [self._sites(a, t, t + 1)[0, PELVIS, :2] for a in range(2)]
        return float(np.linalg.norm(p[0] - p[1]))

    def _blocking(self, agent: int, t: int) -> bool:
        """``agent`` is in guard and facing its opponent at frame ``t``."""
        tr = self._tracks[agent]
        if tr.clip.kind != "guard":
            return False
        to = (
            self._sites(1 - agent, t, t + 1)[0, PELVIS, :2]
            - self._sites(agent, t, t + 1)[0, PELVIS, :2]
        )
        return (
            float(tr.forward @ to / max(np.linalg.norm(to), 1e-9))
            >= self.cfg.block_facing_cos
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def step(
        self, action: int | np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Play ``a0``'s chosen skill to its end (or a KO) against the opponent.

        Args:
            action: Index into ``menu``.
            **kwargs: Ignored compatibility kwargs.

        Returns:
            ``(obs, reward, terminated, truncated, info)``; ``terminated`` on a
            KO, ``truncated`` once the round time is used up.
        """
        cfg = self.cfg
        name = self.menu[int(np.asarray(action).item())]
        kind = self.clips[name].kind
        t = t_start = self._t
        terms = dict.fromkeys(("hit", "hurt", "block", "range", "whiff", "ko"), 0.0)
        if kind in PUNCH_KINDS and self._distance(t) > cfg.whiff_distance_m:
            terms["whiff"] = cfg.r_whiff
        self._schedule(0, name, t)
        end = self._tracks[0].end
        ko = -1
        frames = []
        while t < end and ko < 0:
            if self._tracks[1].end <= t:
                self._schedule(1, self._opponent_skill(t), t)
            stop = min(end, self._tracks[1].end)
            s = [self._sites(a, t, stop) for a in range(2)]
            speed = [
                self._tracks[a].clip.fist_speed[self._tracks[a].frames(t, stop)]
                for a in range(2)
            ]
            dmg = []  # damage dealt by a0, a1
            prevented = np.zeros(stop - t)  # damage a0's block stopped
            for a in range(2):
                d, _, self._in_zone[a] = fist_hit_damage(
                    s[a][:, FISTS],
                    speed[a],
                    s[1 - a][:, HEAD],
                    s[1 - a][:, BODY],
                    self._in_zone[a],
                    cfg.head_radius_m,
                    cfg.body_radius_m,
                    cfg.w_head,
                    cfg.w_body,
                    cfg.hit_min_speed_mps,
                    cfg.hit_full_speed_mps,
                )
                if self._blocking(1 - a, t):
                    if a == 1:
                        prevented = d * (1.0 - cfg.block_factor)
                    d = d * cfg.block_factor
                dmg.append(d)
            punching = [self._tracks[a].clip.kind in PUNCH_KINDS for a in range(2)]
            mask, cut = interrupt_mask(
                np.stack(dmg),
                np.array(punching),
                np.array([self._tracks[a].interrupted for a in range(2)]),
                cfg.interrupt_factor,
            )
            dmg = [dmg[a] * mask[a] for a in range(2)]
            prevented = prevented * mask[1]
            received = self._damage[:, None] + np.cumsum(dmg[::-1], axis=1)
            over = np.flatnonzero((received >= cfg.ko_damage).any(axis=0))
            n = stop - t if over.size == 0 else int(over[0]) + 1
            dist = np.linalg.norm(s[0][:n, PELVIS, :2] - s[1][:n, PELVIS, :2], axis=1)
            terms["hit"] += cfg.r_hit * float(dmg[0][:n].sum())
            terms["hurt"] -= cfg.r_hurt * float(dmg[1][:n].sum())
            terms["block"] += cfg.r_block * float(prevented[:n].sum())
            self._blocked += float(prevented[:n].sum())
            terms["range"] += (
                cfg.r_range * float(in_range(dist, *cfg.range_m).sum()) / self.hz
            )
            self._damage += [float(dmg[1][:n].sum()), float(dmg[0][:n].sum())]
            for a in range(2):  # hits after the KO frame do not count
                self._tracks[a].interrupted |= bool(cut[a] and (mask[a][:n] < 1).any())
            if cfg.record_frames:
                frames.append(self._frames_qpos(t, t + n))
            t += n
            if over.size:
                ko = int(self._damage[0] >= cfg.ko_damage)  # 1: a0 knocked out
        if ko >= 0:
            terms["ko"] = -cfg.r_ko if ko == 1 else cfg.r_ko
        self._t = t
        self._write_pose(t - 1)
        self._last_rwd = {
            "dense": float(sum(terms.values())),
            "done": ko >= 0,
            **{f"r_{k}": v for k, v in terms.items()},
            "skill": name,
            "opponent": self._opponent,
            "distance": self._distance(t - 1),
            "damage_dealt": float(self._damage[1]),
            "damage_received": float(self._damage[0]),
            "damage_blocked": self._blocked,
        }
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs, reward, terminated, _, info = self._finalize_step(
            self._get_obs_dict(self._accessor), self._last_rwd
        )
        if cfg.record_frames:
            info["qpos_frames"] = (
                np.concatenate(frames) if frames else np.zeros((0, 2 * NQ))
            )
        info["frames"] = t - t_start
        truncated = not terminated and t >= self.round_frames
        return obs, reward, terminated, truncated, info

    def _frames_qpos(self, t0: int, t1: int) -> np.ndarray:
        """``(t1 - t0, 178)`` poses of both boxers (a0 then a1) over ``[t0, t1)``."""
        return np.concatenate(
            [self._tracks[a].qpos[self._tracks[a].frames(t0, t1)] for a in range(2)],
            axis=1,
        )

    def _write_pose(self, t: int) -> None:
        """Write both boxers' poses at frame ``t`` into ``data`` and refresh it."""
        q = self._frames_qpos(t, t + 1)[0]
        for a, adr in enumerate(self._qpos_adr):
            self.data.qpos[adr : adr + NQ] = q[a * NQ : (a + 1) * NQ]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _get_obs_dict(self, accessor: CpuEnvAccessor | None) -> dict[str, np.ndarray]:
        """Observation of ``a0`` at the current decision frame (see class docstring)."""
        return self.obs_dict_for(0, self._t)

    def obs_dict_for(self, agent: int, t: int) -> dict[str, np.ndarray]:
        """Observation of boxer ``agent`` deciding at frame ``t``, from its own side.

        Both boxers get the same layout, so one policy can drive either of them.
        """
        own, opp = self._tracks[agent], self._tracks[1 - agent]
        p_own = self._sites(agent, t - 1, t)[0, PELVIS, :2]
        p_opp = self._sites(1 - agent, t - 1, t)[0, PELVIS, :2]
        c, s = own.forward
        to_local = np.array([[c, s], [-s, c]])  # world -> own heading frame
        rel = to_local @ (p_opp - p_own)
        frame = int(np.clip(t - opp.t0, 0, opp.clip.n))
        return {
            "opp_rel": np.array([*rel, np.linalg.norm(rel)]),
            "opp_heading": to_local @ opp.forward,
            "opp_kind": np.eye(len(KINDS))[KINDS.index(opp.clip.kind)],
            "opp_phase": np.array([frame / opp.clip.n, (opp.clip.n - frame) / 100.0]),
            "own_kind": np.eye(len(KINDS))[KINDS.index(own.clip.kind)],
            "damage": self._damage[[agent, 1 - agent]] / self.cfg.ko_damage,
            "time_left": np.array([max(0.0, 1.0 - t / self.round_frames)]),
            "ring_pos": own.qpos[-1, :2] / self.cfg.ring_half_m,
        }

    def _obs_vec(self, agent: int, t: int) -> np.ndarray:
        """Flat float32 observation of boxer ``agent`` at frame ``t``."""
        return self._ensure_obs_gymnasium_compliant(
            self._obs_dict_to_vec(self.obs_dict_for(agent, t))
        )

    # ------------------------------------------------------------------
    # Self-play opponent pool
    # ------------------------------------------------------------------

    def add_opponent(self, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        """Add a frozen policy to the opponent pool (drops the oldest when full).

        Args:
            policy: Maps a boxer's observation to probabilities over ``menu``.
                The env draws the skill from them with its own RNG, so
                ``reset(seed=...)`` still fixes the episode.
        """
        if self.opp_menu != self.menu:
            raise ValueError("self-play needs opponent_menu == menu")
        self._pool = [*self._pool, policy][-self.cfg.pool_size :]

    def clear_opponents(self) -> None:
        """Empty the opponent pool (scripted opponents only)."""
        self._pool = []

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Reward terms of the last decision (computed in :meth:`step`)."""
        return self._last_rwd
