"""Train Midori's goal-conditioned high-level navigator with PPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mujoco"))
from goal_navigation_env import MidoriGoalNavigationEnv


DEFAULT_OUTPUT = ROOT / "training" / "goal_nav_ppo"


def make_env(output: Path, rank: int, seed: int):
    def initialize():
        env = MidoriGoalNavigationEnv()
        monitor_path = output / "monitor" / f"train_{rank}"
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        return Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=("is_success", "is_fallen", "out_of_bounds"),
        )

    return initialize


def train(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env_fns = [make_env(output, rank, args.seed + rank) for rank in range(args.envs)]

    if args.envs == 1:
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns, start_method="spawn")

    eval_env = DummyVecEnv([make_env(output, "eval", args.seed + 10_000)])
    train_env.seed(args.seed)
    eval_env.seed(args.seed + 10_000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output / "best"),
        log_path=str(output / "eval"),
        eval_freq=max(args.eval_freq // args.envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.envs, 1),
        save_path=str(output / "checkpoints"),
        name_prefix="goal_nav_ppo",
        verbose=1,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    if args.resume is not None:
        model = PPO.load(str(args.resume), env=train_env, device="cpu")
        print(f"Resuming checkpoint at {model.num_timesteps:,} existing decisions: {args.resume}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    print(
        f"Training goal PPO for {args.timesteps:,} additional transitions, "
        f"{args.envs} MuJoCo env(s), "
        "CPU policy; actions are stop/forward/backward/left/right."
    )
    print("This learns high-level navigation over the existing crawl gait; it does not fine-tune the 10M joint policy.")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        log_interval=1,
        progress_bar=False,
        reset_num_timesteps=args.resume is None,
    )
    model.save(output / "goal_nav_ppo_final")
    print(f"Saved final policy to {output / 'goal_nav_ppo_final.zip'}")
    best_path = output / "best" / "best_model.zip"
    if best_path.exists():
        print(f"Best evaluation policy: {best_path}")
    else:
        print("No evaluation checkpoint was saved yet; increase timesteps or reduce --eval-freq.")
    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue updating from an existing PPO .zip checkpoint",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (
        args.timesteps < 1
        or args.envs < 1
        or args.eval_freq < 1
        or args.eval_episodes < 1
        or args.checkpoint_freq < 1
    ):
        parser.error("timesteps, envs, eval-freq, eval-episodes, and checkpoint-freq must all be positive")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"resume checkpoint does not exist: {args.resume}")
    train(args)


if __name__ == "__main__":
    main()
