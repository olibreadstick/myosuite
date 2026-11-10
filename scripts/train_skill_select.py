"""Train, evaluate and watch the high-level boxing skill selector (``SkillSelectEnv``).

The low level is kinematic skill playback, so training is fast on CPU::

    # PPO for 200k decisions (a few minutes), saved to logs/skill_select/model.zip
    python scripts/train_skill_select.py train --steps 200000

    # self-play: the opponent is a snapshot of the policy itself 80 % of the time
    python scripts/train_skill_select.py train --self-play --out logs/skill_select/selfplay

    # compare the policy with fixed baselines against each scripted opponent
    # (and against another saved policy)
    python scripts/train_skill_select.py eval logs/skill_select/selfplay.zip \
        --opponent-model logs/skill_select/model_v3.zip

    # watch it box in the MuJoCo viewer (macOS: run with mjpython)
    mjpython scripts/train_skill_select.py view logs/skill_select/model.zip --opponent pressure
    mjpython scripts/train_skill_select.py view logs/skill_select/selfplay.zip \
        --opponent-model logs/skill_select/model_v3.zip
"""

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_vs.skill_select_env import (
    OPPONENTS,
    SkillSelectConfig,
    SkillSelectEnv,
)

OUT = Path("logs/skill_select/model")


def _baselines(env: SkillSelectEnv) -> dict[str, Callable[[], int]]:
    """Fixed reference policies: always guard, uniform random, close in then one-two."""
    menu = env.menu

    def close_then_punch() -> int:
        far = env._distance(env._t - 1) > env.cfg.range_m[1]
        return menu.index("step_fwd" if far else "one_two_push")

    return {
        "guard": lambda: menu.index(env.guard_skill),
        "random": lambda: int(env.np_random.integers(len(menu))),
        "close_then_one_two": close_then_punch,
    }


def run_episodes(
    env: SkillSelectEnv, act: Callable[[np.ndarray], int], episodes: int, seed: int
) -> dict[str, float]:
    """Mean return, damage dealt/received and KO counts over ``episodes``."""
    stats = {
        k: [] for k in ("return", "dealt", "received", "blocked", "ko_won", "ko_lost")
    }
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, info = 0.0, False, {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs))
            total += reward
            done = terminated or truncated
        stats["return"].append(total)
        stats["dealt"].append(info["damage_dealt"])
        stats["received"].append(info["damage_received"])
        stats["blocked"].append(info["damage_blocked"])
        stats["ko_won"].append(info["damage_dealt"] >= env.cfg.ko_damage)
        stats["ko_lost"].append(info["damage_received"] >= env.cfg.ko_damage)
    return {k: float(np.mean(v)) for k, v in stats.items()}


def _load(path: Path):
    from stable_baselines3 import PPO

    return PPO.load(path)


def frozen_policy(policy) -> Callable[[np.ndarray], np.ndarray]:
    """Freeze a copy of an SB3 policy as ``obs -> skill probabilities`` (pool entry)."""
    import torch

    frozen = copy.deepcopy(policy).to("cpu")
    frozen.set_training_mode(False)

    def probs(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dist = frozen.get_distribution(frozen.obs_to_tensor(obs)[0])
            return dist.distribution.probs[0].numpy()

    return probs


def cmd_train(args: argparse.Namespace) -> None:
    """Train PPO on ``SkillSelectEnv`` against the configured opponents."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def factory(rank: int):
        def _init():
            cfg = SkillSelectConfig(
                opponents=tuple(args.opponents), self_play_prob=args.self_play_prob
            )
            env = Monitor(SkillSelectEnv(cfg))
            env.reset(seed=args.seed + rank)
            return env

        return _init

    env = DummyVecEnv([factory(i) for i in range(args.n_envs)])
    callback = None
    if args.self_play:
        from stable_baselines3.common.callbacks import BaseCallback

        class SnapshotPool(BaseCallback):
            """Every ``snapshot_every`` decisions, add a frozen copy to every env's pool."""

            def _add(self) -> None:
                snap = frozen_policy(self.model.policy)
                for e in self.training_env.envs:
                    e.unwrapped.add_opponent(snap)

            def _on_training_start(self) -> None:
                self._next = args.snapshot_every
                self._add()

            def _on_step(self) -> bool:
                if self.num_timesteps >= self._next:
                    self._next += args.snapshot_every
                    self._add()
                return True

        callback = SnapshotPool()
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=256,
        gamma=args.gamma,
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.tensorboard) if args.tensorboard else None,
    )
    start = time.monotonic()
    model.learn(total_timesteps=args.steps, callback=callback)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved {args.out}.zip after {time.monotonic() - start:.0f} s")


def cmd_eval(args: argparse.Namespace) -> None:
    """Print the policy's results next to the baselines, per opponent."""
    model = _load(args.model)
    env = SkillSelectEnv()
    policies = {"policy": lambda obs: int(model.predict(obs, deterministic=True)[0])}
    policies |= {k: (lambda obs, f=f: f()) for k, f in _baselines(env).items()}
    print(
        f"{'opponent':10s}{'policy':20s}{'return':>8s}{'dealt':>8s}{'received':>10s}"
        f"{'blocked':>9s}{'KO won':>8s}{'KO lost':>9s}"
    )
    groups = list(OPPONENTS)
    if args.opponent_model:
        groups.append(args.opponent_model.stem)
    for opponent in groups:
        env.clear_opponents()
        if opponent in OPPONENTS:
            env.cfg.opponents = (opponent,)
        else:  # the saved policy, always
            env.add_opponent(frozen_policy(_load(args.opponent_model).policy))
            env.cfg.self_play_prob = 1.0
        for name, act in policies.items():
            s = run_episodes(env, act, args.episodes, seed=10_000)
            print(
                f"{opponent[:10]:10s}{name:20s}{s['return']:8.2f}{s['dealt']:8.2f}"
                f"{s['received']:10.2f}{s['blocked']:9.2f}{100 * s['ko_won']:7.0f}%"
                f"{100 * s['ko_lost']:8.0f}%"
            )


def cmd_view(args: argparse.Namespace) -> None:
    """Play episodes in the interactive MuJoCo viewer."""
    import mujoco
    import mujoco.viewer

    model = _load(args.model) if args.model else None
    env = SkillSelectEnv(
        SkillSelectConfig(opponents=(args.opponent,), record_frames=True)
    )
    if args.opponent_model:
        env.add_opponent(frozen_policy(_load(args.opponent_model).policy))
        env.cfg.self_play_prob = 1.0
    m, d = env.model, env.data
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            while not done and viewer.is_running():
                action = (
                    int(model.predict(obs, deterministic=True)[0])
                    if model
                    else env.action_space.sample()
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(
                    f"a0 {info['skill']:16s} opp {env._tracks[1].clip.name:16s} "
                    f"dist {info['distance']:.2f}  reward {reward:+.2f}  "
                    f"damage dealt {info['damage_dealt']:.2f} received "
                    f"{info['damage_received']:.2f} blocked {info['damage_blocked']:.2f}"
                )
                for q in info["qpos_frames"]:
                    if not viewer.is_running():
                        return
                    for a, adr in enumerate(env._qpos_adr):
                        d.qpos[adr : adr + 89] = q[a * 89 : (a + 1) * 89]
                    mujoco.mj_forward(m, d)
                    viewer.sync()
                    time.sleep(0.01 / args.speed)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run a subcommand."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="Train PPO.")
    t.add_argument("--steps", type=int, default=200_000, help="Decisions to train for.")
    t.add_argument("--n-envs", type=int, default=8)
    t.add_argument(
        "--n-steps", type=int, default=256, help="Decisions per env per update."
    )
    t.add_argument("--gamma", type=float, default=0.95, help="Discount per decision.")
    t.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=OPPONENTS)
    t.add_argument(
        "--self-play", action="store_true", help="Add policy snapshots as opponents."
    )
    t.add_argument("--self-play-prob", type=float, default=0.8)
    t.add_argument("--snapshot-every", type=int, default=20_000, help="Decisions.")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, default=OUT)
    t.add_argument("--tensorboard", type=Path, default=None)
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval", help="Compare a policy with the baselines.")
    e.add_argument("model", type=Path)
    e.add_argument("--episodes", type=int, default=30)
    e.add_argument(
        "--opponent-model", type=Path, default=None, help="Also play this policy."
    )
    e.set_defaults(func=cmd_eval)
    v = sub.add_parser("view", help="Watch episodes in the MuJoCo viewer.")
    v.add_argument("model", type=Path, nargs="?", default=None, help="Omit for random.")
    v.add_argument("--opponent", default="pressure", choices=OPPONENTS)
    v.add_argument(
        "--opponent-model", type=Path, default=None, help="Blue is this saved policy."
    )
    v.add_argument("--episodes", type=int, default=3)
    v.add_argument("--speed", type=float, default=1.0)
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_view)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
