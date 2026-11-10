"""Watch a trained goal-navigation PPO policy in the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mujoco"))
from goal_navigation_env import MidoriGoalNavigationEnv


DEFAULT_MODEL = ROOT / "training" / "goal_nav_ppo" / "best" / "best_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="run faster than real time (viewer still updates each policy step)",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not args.model.is_file():
        parser.error(f"model checkpoint not found: {args.model}")

    model = PPO.load(str(args.model), device="cpu")
    env = MidoriGoalNavigationEnv(render_mode="human")
    step_seconds = (
        env.frame_skip * env.model.opt.timestep * env.action_repeat
    )
    try:
        print(f"Loaded policy: {args.model.resolve()}")
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            print(
                f"Episode {episode + 1}/{args.episodes}: "
                f"goal=({info['goal_xy_world'][0]:+.2f}, "
                f"{info['goal_xy_world'][1]:+.2f}) m, "
                f"start distance={info['goal_distance_m']:.2f} m"
            )
            deadline = time.monotonic()
            last_command = None

            while env.viewer is not None and env.viewer.is_running():
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)

                if info["command"] != last_command or env.step_count % 10 == 0:
                    print(
                        f"  t={info['episode_seconds']:5.1f}s "
                        f"command={info['command']:8s} "
                        f"distance={info['goal_distance_m']:.2f} m"
                    )
                    last_command = info["command"]

                if not args.fast:
                    deadline += step_seconds
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        deadline = time.monotonic()

                if terminated or truncated:
                    outcome = (
                        "SUCCESS" if info["is_success"] else
                        "FELL" if info["is_fallen"] else
                        "OUT OF BOUNDS" if info["out_of_bounds"] else
                        "TIME LIMIT"
                    )
                    print(
                        f"  {outcome}; final distance="
                        f"{info['goal_distance_m']:.2f} m; "
                        f"simulated={info['episode_seconds']:.1f}s"
                    )
                    break

            if env.viewer is None or not env.viewer.is_running():
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
