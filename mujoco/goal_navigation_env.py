"""Goal-conditioned MuJoCo environment for Midori's command-level navigator."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "hello.xml"
sys.path.insert(0, str(HERE))
import gait

ARENA_HALF_SIZE_M = 1.5
ARENA_TERMINATION_LIMIT_M = 1.42
GOAL_SAMPLING_LIMIT_M = 1.20
GOAL_SAMPLE_MIN_RADIUS_M = 0.45
GOAL_SAMPLE_MAX_RADIUS_M = 1.15
GOAL_TOUCH_RADIUS_M = 0.06
ROOT_STAND_HEIGHT_M = 0.1354
REST_FEMUR_ANGLE_RAD = -0.55
REST_TIBIA_ANGLE_RAD = 1.10
JOINT_VELOCITY_SCALE_RAD_S = 12.0
BASE_LINEAR_VELOCITY_SCALE_M_S = 1.0
BASE_ANGULAR_VELOCITY_SCALE_RAD_S = 4.0

RL_COMMANDS = ("stop", "forward", "backward", "left", "right")
OBSERVATION_SIZE = 59
OBSERVATION_LAYOUT = (
    "goal_dx_body_normalized", "goal_dy_body_normalized",
    "goal_distance_normalized", "sin_goal_bearing_body", "cos_goal_bearing_body",
    "gravity_x_body", "gravity_y_body", "gravity_z_body",
    "linear_velocity_x_body", "linear_velocity_y_body", "linear_velocity_z_body",
    "angular_velocity_x_body", "angular_velocity_y_body", "angular_velocity_z_body",
    *(f"joint_{i}_position_normalized" for i in range(12)),
    *(f"joint_{i}_velocity_normalized" for i in range(12)),
    *(f"foot_{i}_{axis}_body_normalized" for i in range(4) for axis in "xyz"),
    *(f"foot_{i}_ground_contact" for i in range(4)),
    *(f"active_command_{name}" for name in RL_COMMANDS),
)


class MidoriGoalNavigationEnv(gym.Env):
    """Goal-reaching task with randomized goals and a fixed crawl executor.

    ``reset(options={"goal_xy": (x, y)})`` pins a goal for evaluation or
    deployment. With no explicit goal, reset samples a random reachable point.
    The returned observation contains only body-relative goal information,
    robot state, foot contacts, and the previous action; absolute world pose is
    intentionally kept out of the policy input.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        frame_skip: int = 10,
        episode_seconds: float = 60.0,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)
        self.action_repeat = 10
        self.max_episode_steps = max(
            1,
            int(episode_seconds / (self.model.opt.timestep * self.frame_skip * self.action_repeat)),
        )

        self.root_joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos_address = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_dof_address = int(self.model.jnt_dofadr[self.root_joint_id])
        self.chassis_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.floor_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.goal_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "rl_goal_marker")
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body_id])
        if self.goal_mocap_id < 0:
            raise RuntimeError("rl_goal_marker must be a mocap body in the MuJoCo XML")

        self.foot_names = ("front_left", "front_right", "rear_left", "rear_right")
        self.foot_geom_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, f"{name}_foot")
            for name in self.foot_names
        ]
        self.foot_geom_id_by_leg = dict(zip(self.foot_names, self.foot_geom_ids))
        self.joint_names = [name for _, name in gait.JOINTS]
        self.joint_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        }
        self.actuator_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            for name in self.joint_names
        }
        self.actuator_joint_ids = [
            self.joint_ids[name] for name in self.joint_names
        ]
        if len(self.actuator_joint_ids) != 12:
            raise RuntimeError(f"Expected 12 servo actuators, found {len(self.actuator_joint_ids)}")
        self.joint_qpos_addresses = [
            int(self.model.jnt_qposadr[joint_id]) for joint_id in self.actuator_joint_ids
        ]
        self.joint_dof_addresses = [
            int(self.model.jnt_dofadr[joint_id]) for joint_id in self.actuator_joint_ids
        ]
        self.actuator_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32).copy()
        self.actuator_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32).copy()
        self.joint_low = np.array(
            [self.model.jnt_range[joint_id, 0] for joint_id in self.actuator_joint_ids],
            dtype=np.float32,
        )
        self.joint_high = np.array(
            [self.model.jnt_range[joint_id, 1] for joint_id in self.actuator_joint_ids],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(RL_COMMANDS))
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        self.goal_xy = np.zeros(2, dtype=np.float32)
        self.previous_distance = 0.0
        self.current_command = "stop"
        self.gait_base_pose = {}
        self.prepare_started_at = None
        self.prepare_from_pose = None
        self.gait_started_at = 0.0
        self.blend_start = None
        self.last_targets = {}
        self.stance_corrections = {leg: 0.0 for leg in gait.LEG_INFO}
        self.last_foot_contacts = {leg: False for leg in gait.LEG_INFO}
        self.step_count = 0
        self.viewer = None

    def _require_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"Missing MuJoCo object {name!r} in {self.model_path}")
        return int(object_id)

    def _sample_goal(self, start_xy: np.ndarray) -> np.ndarray:


        radius = self.np_random.uniform(
            GOAL_SAMPLE_MIN_RADIUS_M, GOAL_SAMPLE_MAX_RADIUS_M
        )
        bearing = self.np_random.uniform(-math.pi, math.pi)
        candidate = start_xy + radius * np.array(
            [math.cos(bearing), math.sin(bearing)], dtype=np.float32
        )
        if np.any(np.abs(candidate) > GOAL_SAMPLING_LIMIT_M):


            for _ in range(100):
                bearing = self.np_random.uniform(-math.pi, math.pi)
                candidate = start_xy + radius * np.array(
                    [math.cos(bearing), math.sin(bearing)], dtype=np.float32
                )
                if np.all(np.abs(candidate) <= GOAL_SAMPLING_LIMIT_M):
                    break
            else:
                raise RuntimeError("Could not sample a goal inside the arena")
        return candidate.astype(np.float32)

    def _body_rotation(self) -> np.ndarray:
        return self.data.xmat[self.chassis_body_id].reshape(3, 3)

    def _goal_relative_body(self) -> np.ndarray:
        root_xy = self.data.qpos[self.root_qpos_address:self.root_qpos_address + 2]
        goal_world = np.array(
            [self.goal_xy[0] - root_xy[0], self.goal_xy[1] - root_xy[1], 0.0],
            dtype=np.float64,
        )
        return self._body_rotation().T @ goal_world

    def _distance_to_goal(self) -> float:
        delta = self.goal_xy - self.data.qpos[
            self.root_qpos_address:self.root_qpos_address + 2
        ]
        return float(np.linalg.norm(delta))

    def _goal_touched(self) -> bool:
        for geom_id in self.foot_geom_ids:
            foot = self.data.geom_xpos[geom_id]
            if foot[2] <= 0.04 and np.linalg.norm(foot[:2] - self.goal_xy) <= GOAL_TOUCH_RADIUS_M:
                return True
        return False

    def _fallen(self) -> bool:
        rotation = self._body_rotation()
        return bool(
            self.data.qpos[self.root_qpos_address + 2] < 0.055
            or rotation[2, 2] < 0.35
        )

    def _out_of_bounds(self) -> bool:
        x, y = self.data.qpos[self.root_qpos_address:self.root_qpos_address + 2]
        return bool(
            abs(float(x)) > ARENA_TERMINATION_LIMIT_M
            or abs(float(y)) > ARENA_TERMINATION_LIMIT_M
        )

    def _foot_contacts(self) -> np.ndarray:
        flags = np.zeros(4, dtype=np.float32)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            for index, foot_geom_id in enumerate(self.foot_geom_ids):
                if {contact.geom1, contact.geom2} == {foot_geom_id, self.floor_geom_id}:
                    flags[index] = 1.0
        return flags

    def _select_command(self, command: str) -> None:
        previous = self.current_command
        if command == previous:
            return
        self.current_command = command
        self.stance_corrections = {leg: 0.0 for leg in gait.LEG_INFO}
        if previous == "stop" and command != "stop":
            self.prepare_started_at = float(self.data.time)
            self.prepare_from_pose = {
                name: float(self.data.qpos[self.joint_qpos_addresses[index]])
                for index, name in enumerate(self.joint_names)
            }
            self.gait_base_pose = dict(self.prepare_from_pose)
            for name in self.joint_names:
                if name.endswith("hip_pitch"):
                    self.gait_base_pose[name] = gait.WALK_STANCE_FEMUR_ANGLE
                elif name.endswith("knee_pitch"):
                    self.gait_base_pose[name] = gait.WALK_STANCE_TIBIA_ANGLE
            self.gait_started_at = self.data.time + gait.WALK_STANCE_TRANSITION_SECONDS
            self.blend_start = self.gait_started_at
        elif command != "stop":
            self.prepare_started_at = None
            self.prepare_from_pose = None
            self.gait_started_at = float(self.data.time)
            self.blend_start = None
        else:
            self.prepare_started_at = None
            self.prepare_from_pose = None
            self.blend_start = None

    def _joint_targets(self) -> dict[str, float]:
        if self.current_command == "stop":
            return gait.crawl_targets(0.0, "stop")
        if self.prepare_started_at is not None:
            progress = min(
                1.0,
                max(0.0, self.data.time - self.prepare_started_at)
                / gait.WALK_STANCE_TRANSITION_SECONDS,
            )
            blend = gait.smoothstep01(progress)
            targets = {
                name: self.prepare_from_pose[name]
                + blend * (self.gait_base_pose[name] - self.prepare_from_pose[name])
                for name in self.joint_names
            }
            if progress >= 1.0:
                self.prepare_started_at = None
                self.prepare_from_pose = None
            return targets

        gait_time = max(0.0, self.data.time - self.gait_started_at)
        swinging_legs = gait.crawl_swinging_legs(gait_time)
        for leg in gait.LEG_INFO:
            if leg in swinging_legs:
                self.stance_corrections[leg] = 0.0
            elif self.last_foot_contacts[leg]:
                self.stance_corrections[leg] = max(
                    0.0,
                    self.stance_corrections[leg]
                    - gait.STANCE_RELEASE_RATE_MPS * self.model.opt.timestep,
                )
            else:
                self.stance_corrections[leg] = min(
                    gait.STANCE_SEARCH_MAX_METRES,
                    self.stance_corrections[leg]
                    + gait.STANCE_SEARCH_RATE_MPS * self.model.opt.timestep,
                )
        blend = 1.0
        if self.blend_start is not None:
            progress = min(
                1.0,
                max(0.0, self.data.time - self.blend_start)
                / gait.WALK_GAIT_BLEND_SECONDS,
            )
            blend = gait.smoothstep01(progress)
            if progress >= 1.0:
                self.blend_start = None
        return gait.crawl_targets(
            gait_time,
            self.current_command,
            self.gait_base_pose,
            blend,
            self.stance_corrections,
        )

    def _observation(self) -> np.ndarray:
        rotation = self._body_rotation()
        goal_rel = self._goal_relative_body()
        goal_distance = float(np.linalg.norm(goal_rel[:2]))
        bearing = math.atan2(float(goal_rel[1]), float(goal_rel[0]))
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])

        root_dof = self.root_dof_address
        linear_velocity_body = rotation.T @ self.data.qvel[root_dof:root_dof + 3]
        angular_velocity_body = rotation.T @ self.data.qvel[root_dof + 3:root_dof + 6]

        joint_positions = np.array(
            [self.data.qpos[address] for address in self.joint_qpos_addresses], dtype=np.float32
        )
        joint_velocities = np.array(
            [self.data.qvel[address] for address in self.joint_dof_addresses], dtype=np.float32
        )
        joint_position_norm = 2.0 * (joint_positions - self.joint_low) / np.maximum(
            self.joint_high - self.joint_low, 1e-6
        ) - 1.0
        joint_velocity_norm = np.clip(
            joint_velocities / JOINT_VELOCITY_SCALE_RAD_S, -1.0, 1.0
        )

        root_position = self.data.xpos[self.chassis_body_id]
        foot_positions = []
        for geom_id in self.foot_geom_ids:
            foot_world = self.data.geom_xpos[geom_id] - root_position
            foot_body = rotation.T @ foot_world
            foot_positions.extend(np.clip(foot_body / ARENA_HALF_SIZE_M, -1.0, 1.0))

        observation = np.concatenate((
            np.array([
                goal_rel[0] / ARENA_HALF_SIZE_M,
                goal_rel[1] / ARENA_HALF_SIZE_M,
                min(goal_distance / (math.sqrt(2.0) * ARENA_HALF_SIZE_M), 1.0),
                math.sin(bearing),
                math.cos(bearing),
            ], dtype=np.float32),
            np.asarray(gravity_body, dtype=np.float32),
            np.clip(linear_velocity_body / BASE_LINEAR_VELOCITY_SCALE_M_S, -5.0, 5.0),
            np.clip(angular_velocity_body / BASE_ANGULAR_VELOCITY_SCALE_RAD_S, -5.0, 5.0),
            np.clip(joint_position_norm, -1.0, 1.0),
            joint_velocity_norm,
            np.asarray(foot_positions, dtype=np.float32),
            self._foot_contacts(),
            np.eye(len(RL_COMMANDS), dtype=np.float32)[RL_COMMANDS.index(self.current_command)],
        )).astype(np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Observation shape mismatch: expected 59, got {observation.shape}")
        return observation.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)

        start_xy = np.asarray(
            options.get("start_xy", self.np_random.uniform(-0.12, 0.12, size=2)),
            dtype=np.float64,
        )
        if start_xy.shape != (2,) or np.any(np.abs(start_xy) > 0.3):
            raise ValueError("start_xy must contain two coordinates within 0.3 m of arena center")
        yaw = float(options.get("start_yaw", self.np_random.uniform(-math.pi, math.pi)))
        qadr = self.root_qpos_address
        self.data.qpos[qadr:qadr + 3] = [start_xy[0], start_xy[1], ROOT_STAND_HEIGHT_M]
        self.data.qpos[qadr + 3:qadr + 7] = [
            math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
        ]

        for actuator_index, joint_id in enumerate(self.actuator_joint_ids):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name.endswith("hip_pitch"):
                angle = REST_FEMUR_ANGLE_RAD
            elif joint_name.endswith("knee_pitch"):
                angle = REST_TIBIA_ANGLE_RAD
            else:
                angle = 0.0
            self.data.qpos[self.joint_qpos_addresses[actuator_index]] = angle
            self.data.ctrl[actuator_index] = angle

        if "goal_xy" in options:
            goal_xy = np.asarray(options["goal_xy"], dtype=np.float32)
            if goal_xy.shape != (2,) or np.any(np.abs(goal_xy) > GOAL_SAMPLING_LIMIT_M):
                raise ValueError("goal_xy must contain two coordinates within the 1.2 m goal area")
            if np.linalg.norm(goal_xy - start_xy) < 0.25:
                raise ValueError("goal_xy must be at least 0.25 m from start_xy")
            self.goal_xy = goal_xy
        else:
            self.goal_xy = self._sample_goal(start_xy.astype(np.float32))

        self.data.mocap_pos[self.goal_mocap_id] = [self.goal_xy[0], self.goal_xy[1], 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.previous_distance = self._distance_to_goal()
        self.current_command = "stop"
        self.gait_base_pose = {
            name: float(self.data.qpos[self.joint_qpos_addresses[index]])
            for index, name in enumerate(self.joint_names)
        }
        self.prepare_started_at = None
        self.prepare_from_pose = None
        self.gait_started_at = float(self.data.time)
        self.blend_start = None
        self.last_targets = dict(self.gait_base_pose)
        self.stance_corrections = {leg: 0.0 for leg in gait.LEG_INFO}
        self.last_foot_contacts = gait.foot_contact_flags(
            self.data, self.foot_geom_id_by_leg, self.floor_geom_id
        )
        self.step_count = 0
        observation = self._observation()
        info = {
            "goal_xy_world": self.goal_xy.copy(),
            "goal_relative_xy_body_m": self._goal_relative_body()[:2].astype(np.float32),
            "goal_distance_m": self.previous_distance,
        }
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        command_index = int(np.asarray(action).item())
        if command_index < 0 or command_index >= len(RL_COMMANDS):
            raise ValueError(f"Action must be an integer from 0 to {len(RL_COMMANDS) - 1}")
        command = RL_COMMANDS[command_index]
        switched_command = command != self.current_command
        self._select_command(command)
        for _ in range(self.action_repeat):
            targets = self._joint_targets()
            self.last_targets = dict(targets)
            for name, target in targets.items():
                self.data.ctrl[self.actuator_ids[name]] = target
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            self.last_foot_contacts = gait.foot_contact_flags(
                self.data, self.foot_geom_id_by_leg, self.floor_geom_id
            )

        self.step_count += 1
        distance = self._distance_to_goal()
        progress_reward = 4.0 * (self.previous_distance - distance)
        living_cost = -0.01
        switch_cost = -0.005 if switched_command else 0.0
        reward = progress_reward + living_cost + switch_cost
        self.previous_distance = distance

        fallen = self._fallen()
        out_of_bounds = self._out_of_bounds()
        success = self._goal_touched() and not fallen and not out_of_bounds
        terminated = success or fallen or out_of_bounds
        truncated = self.step_count >= self.max_episode_steps and not terminated
        if success:
            reward += 25.0
        elif fallen:
            reward -= 25.0
        elif out_of_bounds:
            reward -= 10.0

        observation = self._observation()
        info = {
            "is_success": success,
            "is_fallen": fallen,
            "out_of_bounds": out_of_bounds,
            "goal_xy_world": self.goal_xy.copy(),
            "goal_relative_xy_body_m": self._goal_relative_body()[:2].astype(np.float32),
            "goal_distance_m": distance,
            "episode_seconds": (
                self.step_count * self.action_repeat * self.frame_skip * self.model.opt.timestep
            ),
            "reward_progress": progress_reward,
            "reward_living_cost": living_cost,
            "reward_command_switch": switch_cost,
            "command": command,
        }
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -40
            self.viewer.cam.distance = 2.4
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        if self.viewer.is_running():
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
