"""Write simulation episodes and their state/action CSV files."""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from gait import (
    DATASET_SAMPLE_PERIOD_SECONDS,
    FOOT_LIFT_METRES,
    GAIT_CONTROLLER_VERSION,
    GAIT_FREQUENCY_HZ,
    SIM_SERVO_KP,
    SIM_SERVO_KV,
    SIM_SERVO_TORQUE_LIMIT_NM,
    STANCE_PRELOAD_METRES,
    STANCE_SEARCH_MAX_METRES,
    STANCE_SEARCH_RATE_MPS,
    STRIDE_METRES,
    SUPPORT_SHIFT_FRACTION,
    SWING_FRACTION,
    TURN_STRIDE_METRES,
    WALK_GAIT_BLEND_SECONDS,
    WALK_STANCE_FEMUR_ANGLE,
    WALK_STANCE_TIBIA_ANGLE,
    WALK_STANCE_TRANSITION_SECONDS,
)


LOG_DIR = Path(__file__).resolve().parent / "logs"


class EpisodeRecorder:
    """Write one simulation-time CSV and metadata file per labeled episode."""

    def __init__(self, joint_names, leg_names):
        self.joint_names = joint_names
        self.leg_names = leg_names
        self.active = False
        self.row_count = 0
        self.episode_dir = None
        self.file = None
        self.writer = None
        self.metadata = None
        self.start_sim_time = None

    def start(self, task, sim_time, model, balance_assist_enabled):
        if self.active:
            raise RuntimeError("An episode is already being recorded")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.episode_id = f"sim_{stamp}"
        self.episode_dir = LOG_DIR / "episodes" / self.episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.start_sim_time = sim_time
        self.row_count = 0
        self.file = (self.episode_dir / "state_action.csv").open(
            "w", newline="", encoding="utf-8"
        )
        columns = [
            "episode_id", "step_index", "episode_time_s", "sim_time_s",
            "recorded_at_local", "movement_mode", "balance_assist_active",
            "applied_controller", "target_label_source",
            "root_x_m", "root_y_m", "root_z_m",
            "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
            "root_vx_mps", "root_vy_mps", "root_vz_mps",
            "root_wx_radps", "root_wy_radps", "root_wz_radps",
        ]
        columns += [f"target_{name}_deg" for name in self.joint_names]
        columns += [f"actual_{name}_deg" for name in self.joint_names]
        columns += [f"actual_{name}_vel_degps" for name in self.joint_names]
        columns += [
            f"foot_{leg}_{axis}_m"
            for leg in self.leg_names for axis in ("x", "y", "z")
        ]
        columns += [f"foot_{leg}_contact" for leg in self.leg_names]
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()
        self.metadata = {
            "schema_version": 3,
            "episode_id": self.episode_id,
            "task": task,
            "source": "mujoco_simulation",
            "hardware_feedback": False,
            "status": "recording",
            "started_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "start_sim_time_s": sim_time,
            "simulation_timestep_s": float(model.opt.timestep),
            "sample_period_sim_s": DATASET_SAMPLE_PERIOD_SECONDS,
            "nominal_sample_rate_hz": 1.0 / DATASET_SAMPLE_PERIOD_SECONDS,
            "simulation_seed": None,
            "environment_randomization": "none",
            "manual_pose_balance_assist_enabled": bool(balance_assist_enabled),
            "gravity_mps2": [float(value) for value in model.opt.gravity],
            "model_file": "mujoco/hello.xml",
            "controller_file": "mujoco/control_test.py",
            "gait_controller_version": GAIT_CONTROLLER_VERSION,
            "gait_parameters": {
                "cycle_frequency_hz": GAIT_FREQUENCY_HZ,
                "forward_stride_m": STRIDE_METRES,
                "turn_stride_m": TURN_STRIDE_METRES,
                "walk_stance_femur_deg": math.degrees(WALK_STANCE_FEMUR_ANGLE),
                "walk_stance_tibia_deg": math.degrees(WALK_STANCE_TIBIA_ANGLE),
                "walk_stance_transition_s": WALK_STANCE_TRANSITION_SECONDS,
                "walk_gait_blend_s": WALK_GAIT_BLEND_SECONDS,
                "swing_fraction": SWING_FRACTION,
                "foot_lift_m": FOOT_LIFT_METRES,
                "stance_preload_m": STANCE_PRELOAD_METRES,
                "support_shift_fraction": SUPPORT_SHIFT_FRACTION,
                "stance_search_rate_mps": STANCE_SEARCH_RATE_MPS,
                "stance_search_max_m": STANCE_SEARCH_MAX_METRES,
                "foot_trajectory": "quintic_eased_swing_and_stance",
            },
            "simulated_servo_parameters": {
                "position_kp": SIM_SERVO_KP,
                "position_kv": SIM_SERVO_KV,
                "torque_limit_nm": SIM_SERVO_TORQUE_LIMIT_NM,
                "hardware_calibrated": False,
            },
            "joint_order": self.joint_names,
            "coordinate_frame": "X forward, Y left, Z up; world frame",
            "outcome": None,
            "command_events": [],
        }
        self._write_metadata()
        self.active = True
        return self.episode_id

    def add_command_event(self, sim_time, event, mode):
        if self.active:
            self.metadata["command_events"].append({
                "sim_time_s": float(sim_time),
                "episode_time_s": float(sim_time - self.start_sim_time),
                "event": event,
                "movement_mode": mode,
            })

    def write(self, row):
        if self.active:
            row["episode_id"] = self.episode_id
            row["step_index"] = self.row_count
            row["episode_time_s"] = row["sim_time_s"] - self.start_sim_time
            self.writer.writerow(row)
            self.file.flush()
            self.row_count += 1

    def _write_metadata(self):
        path = self.episode_dir / "episode.json"
        path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def finish(self, sim_time, outcome="incomplete"):
        if not self.active:
            return None
        self.metadata.update({
            "status": "incomplete" if outcome in ("incomplete", "interrupted") else "complete",
            "ended_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "end_sim_time_s": sim_time,
            "duration_sim_s": sim_time - self.start_sim_time,
            "row_count": self.row_count,
            "outcome": outcome,
        })
        self.file.close()
        self.file = None
        self._write_metadata()
        episode_dir = self.episode_dir
        self.active = False
        return episode_dir


