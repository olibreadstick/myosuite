"""Checks for the kinematic high-level boxing env (SkillSelectEnv) and its hit term."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.envs.myo.tasks.challenge.boxing_vs.skill_library import (
    SITE_KEYS,
    SKILL_DIR,
)
from myosuite.terms.multiplayer.boxing_skill_reward import (
    fist_hit_damage,
    interrupt_mask,
)

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.skipif(
        not SKILL_DIR.is_dir(), reason="boxing skill clips not checked out"
    ),
]


@pytest.fixture(scope="module")
def env():
    from myosuite.envs.myo.tasks.challenge.boxing_vs.skill_select_env import (
        SkillSelectEnv,
    )

    return SkillSelectEnv()


def _rollout(env, seed: int, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    traj, rewards = [obs], []
    for _ in range(n):
        obs, reward, terminated, truncated, _ = env.step(
            int(rng.integers(env.action_space.n))
        )
        traj.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            break
    return np.array(traj), np.array(rewards)


def test_reset_seed_determines_episode(env) -> None:
    a, b = _rollout(env, 3), _rollout(env, 3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(_rollout(env, 4)[0], a[0])


def test_obs_contract(env) -> None:
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        obs, *_ = env.step(env.action_space.sample())


def test_precomputed_sites_match_forward_kinematics(env) -> None:
    env.reset(seed=1)
    for action in range(env.action_space.n):
        env.step(action)
        for a, agent in enumerate(("agent_0", "agent_1")):
            ids = [env.meta.site_ids[agent][k] for k in SITE_KEYS]
            predicted = env._sites(a, env._t - 1, env._t)[0]
            np.testing.assert_allclose(env.data.site_xpos[ids], predicted, atol=1e-6)


def test_punch_lands_in_range_and_whiffs_far(env) -> None:
    env.cfg.opponents, env.cfg.block_factor = ("guard",), 1.0
    punch = env.menu.index("punch_R_push1")
    try:
        env.cfg.separation_m = (0.5, 0.5)
        env.reset(seed=0)
        _, _, _, _, info = env.step(punch)
        assert info["damage_dealt"] > 0.5 and info["r_whiff"] == 0.0
        env.cfg.separation_m = (1.5, 1.5)
        env.reset(seed=0)
        _, _, _, _, info = env.step(punch)
        assert info["damage_dealt"] == 0.0 and info["r_whiff"] < 0.0
    finally:
        env.cfg.opponents, env.cfg.block_factor = ("guard", "random", "pressure"), 0.3
        env.cfg.separation_m = (1.2, 2.0)


def test_steps_never_walk_through_the_opponent(env) -> None:
    env.cfg.opponents = ("pressure",)
    try:
        for seed in range(5):
            env.reset(seed=seed)
            done = False
            while not done:
                *_, terminated, truncated, info = env.step(env.menu.index("step_fwd"))
                assert info["distance"] >= env.cfg.min_separation_m - 1e-6
                own, opp = (
                    env._sites(a, env._t - 1, env._t)[0, -1, :2] for a in (0, 1)
                )
                assert (opp - own) @ env._tracks[0].forward > 0.0  # still in front
                done = terminated or truncated
    finally:
        env.cfg.opponents = ("guard", "random", "pressure")


def test_hit_scores_once_per_entry() -> None:
    fists = np.zeros((10, 2, 3))
    fists[:, 1] = 10.0  # left fist far away
    fists[:, 0, 0] = [0.5, 0.3, 0.1, 0.0, 0.05, 0.1, 0.1, 0.3, 0.1, 0.1]
    speed = np.full((10, 2), 5.0)
    head, body = np.zeros((10, 3)), np.full((10, 3), 5.0)
    args = (0.15, 0.18, 2.0, 1.0, 1.5, 4.5)
    damage, head_hit, last = fist_hit_damage(
        fists, speed, head, body, np.zeros(2, bool), *args
    )
    np.testing.assert_allclose(damage, [0, 0, 2.0, 0, 0, 0, 0, 0, 2.0 / 3.0, 0])
    assert head_hit.sum() == 2 and last.tolist() == [True, False]
    slow = fist_hit_damage(
        fists, np.zeros((10, 2)), head, body, np.zeros(2, bool), *args
    )[0]
    assert not slow.any()


def test_guard_blocks_and_is_rewarded(env) -> None:
    env.cfg.opponents, env.cfg.separation_m = ("pressure",), (0.5, 0.5)
    try:
        env.reset(seed=2)
        block_reward, info = 0.0, {}
        for _ in range(20):
            *_, info = env.step(env.menu.index(env.guard_skill))
            block_reward += info["r_block"]
        assert info["damage_blocked"] > 0.0
        np.testing.assert_allclose(
            block_reward, env.cfg.r_block * info["damage_blocked"]
        )
        # A block lets through block_factor of each hit, so received <= that share.
        share = env.cfg.block_factor / (1.0 - env.cfg.block_factor)
        assert info["damage_received"] <= share * info["damage_blocked"] + 1e-9
    finally:
        env.cfg.opponents = ("guard", "random", "pressure")
        env.cfg.separation_m = (1.2, 2.0)


def test_interrupt_cuts_the_rest_of_a_punch() -> None:
    # a0 lands on frame 3 while a1 is mid-punch: a1's later hit on frame 6 is cut.
    damage = np.zeros((2, 10))
    damage[0, 3], damage[1, 6] = 1.0, 2.0
    mask, cut = interrupt_mask(damage, np.array([True, True]), np.zeros(2, bool), 0.0)
    assert (damage * mask)[1, 6] == 0.0 and (damage * mask)[0, 3] == 1.0
    assert cut.tolist() == [False, True]
    # Same-frame hits are a trade: both land.
    damage[1, 3], damage[1, 6] = 2.0, 0.0
    mask, _ = interrupt_mask(damage, np.array([True, True]), np.zeros(2, bool), 0.0)
    np.testing.assert_array_equal(damage * mask, damage)
    # A cut punch no longer interrupts with its (removed) later hit.
    damage = np.zeros((2, 10))
    damage[0, 2], damage[1, 5], damage[0, 8] = 1.0, 1.0, 1.0
    mask, cut = interrupt_mask(damage, np.array([True, True]), np.zeros(2, bool), 0.0)
    assert (damage * mask)[0, 8] == 1.0 and cut.tolist() == [False, True]
    # Not punching (e.g. guard): never interrupted; already interrupted: cut from frame 0.
    mask, _ = interrupt_mask(
        damage, np.array([True, False]), np.array([True, False]), 0.5
    )
    np.testing.assert_array_equal(mask[0], np.full(10, 0.5))
    np.testing.assert_array_equal(mask[1], np.ones(10))


def test_short_guard_is_on_the_menu(env) -> None:
    assert env.guard_skill == "guard_short"
    assert env.clips["guard_short"].n == env.clips["guard"].n // 2


def test_both_boxers_see_the_same_thing_at_reset(env) -> None:
    env.reset(seed=5)
    a0, a1 = env.obs_dict_for(0, env._t), env.obs_dict_for(1, env._t)
    for key in a0:
        if key == "ring_pos":  # spawned at mirrored positions
            np.testing.assert_allclose(a1[key], -a0[key], atol=1e-6)
        else:
            np.testing.assert_allclose(a1[key], a0[key], atol=1e-6, err_msg=key)
    env._damage[:] = [1.0, 3.0]  # received by a0, a1: each sees (own, opponent's)
    np.testing.assert_allclose(env.obs_dict_for(1, env._t)["damage"], [0.3, 0.1])


def test_pool_opponent_follows_its_policy_and_the_seed(env) -> None:
    punch = env.menu.index("punch_R_stand1")
    onehot = np.eye(env.action_space.n)[punch]
    env.cfg.self_play_prob = 1.0
    try:
        env.add_opponent(lambda obs: onehot)
        env.cfg.separation_m = (0.5, 0.5)
        env.reset(seed=0)
        assert env._opponent == "self"
        for _ in range(4):
            env.step(env.menu.index(env.guard_skill))
            assert env._tracks[1].clip.name == "punch_R_stand1"
        env.clear_opponents()
        env.add_opponent(lambda obs: np.full(env.action_space.n, 1.0))  # uniform
        a, b = _rollout(env, 7), _rollout(env, 7)
        np.testing.assert_array_equal(a[0], b[0])
    finally:
        env.clear_opponents()
        env.cfg.self_play_prob = 0.8
        env.cfg.separation_m = (1.2, 2.0)
