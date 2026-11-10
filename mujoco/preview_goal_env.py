"""Open the goal-navigation arena in the MuJoCo viewer without an RL agent."""

import argparse
import time

from goal_navigation_env import MidoriGoalNavigationEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="repeat the same randomized goal")
    args = parser.parse_args()

    env = MidoriGoalNavigationEnv(render_mode="human")
    try:
        _, info = env.reset(seed=args.seed)
        action = 0

        def show_goal(goal_info):
            goal = goal_info["goal_xy_world"]
            relative = goal_info["goal_relative_xy_body_m"]
            print(
                f"Goal world XY=({goal[0]:+.2f}, {goal[1]:+.2f}) m; "
                f"body-relative XY=({relative[0]:+.2f}, {relative[1]:+.2f}) m; "
                f"distance={goal_info['goal_distance_m']:.2f} m"
            )

        show_goal(info)
        step_seconds = env.frame_skip * env.model.opt.timestep
        deadline = time.monotonic()
        while env.viewer is not None and env.viewer.is_running():
            _, _, terminated, truncated, info = env.step(action)
            deadline += step_seconds
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
            if terminated or truncated:
                _, info = env.reset()
                show_goal(info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
