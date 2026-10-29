"""MuJoCo controller for the 12-servo spider prototype.

Run from the repository root with: uv run python mujoco/control_test.py
Use the episode-capture panel to save labeled demonstrations at 50 Hz simulation time.
"""
import csv
import json
import math
import tkinter as tk
import time
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "hello.xml"
LOG_DIR = HERE / "logs"
DATASET_SAMPLE_PERIOD_SECONDS = 0.02  # 50 Hz in simulation time
GAIT_FREQUENCY_HZ = 0.85
STRIDE_METRES = 0.040
TURN_STRIDE_METRES = 0.040
SWING_FRACTION = 0.25
FOOT_LIFT_METRES = 0.030
STANCE_PRELOAD_METRES = 0.002
SUPPORT_SHIFT_FRACTION = 0.70
STANCE_SEARCH_RATE_MPS = 0.040
STANCE_SEARCH_MAX_METRES = 0.008
STANCE_RELEASE_RATE_MPS = 0.080
GAIT_CONTROLLER_VERSION = "stance_transition_sequential_crawl_v3"
WALK_STANCE_FEMUR_ANGLE = math.radians(-45.0)
WALK_STANCE_TIBIA_ANGLE = math.radians(65.0)
WALK_STANCE_TRANSITION_SECONDS = 1.2
WALK_GAIT_BLEND_SECONDS = 0.35
SIM_SERVO_KP = 10.0
SIM_SERVO_KV = 0.5
SIM_SERVO_TORQUE_LIMIT_NM = 0.45
BALANCE_POSITION_KP = (220.0, 220.0, 420.0)
BALANCE_POSITION_KD = (28.0, 28.0, 38.0)
BALANCE_ROTATION_KP = 5.0
BALANCE_ROTATION_KD = 0.9
BALANCE_FORCE_LIMIT_NEWTONS = 18.0
BALANCE_TORQUE_LIMIT_NM = 2.5

JOINTS = [
    ("Front left coxa yaw", "front_left_hip_abduction"),
    ("Front left femur pitch", "front_left_hip_pitch"),
    ("Front left tibia pitch", "front_left_knee_pitch"),
    ("Front right coxa yaw", "front_right_hip_abduction"),
    ("Front right femur pitch", "front_right_hip_pitch"),
    ("Front right tibia pitch", "front_right_knee_pitch"),
    ("Rear left coxa yaw", "rear_left_hip_abduction"),
    ("Rear left femur pitch", "rear_left_hip_pitch"),
    ("Rear left tibia pitch", "rear_left_knee_pitch"),
    ("Rear right coxa yaw", "rear_right_hip_abduction"),
    ("Rear right femur pitch", "rear_right_hip_pitch"),
    ("Rear right tibia pitch", "rear_right_knee_pitch"),
]

# Leg angle is the direction the coxa points in the chassis XY plane. The
# quarter-cycle offsets produce a crawl sequence with one leg swinging at a time.
LEG_INFO = {
    "front_left": (0.032, 0.027, math.pi / 4, 0.00),
    "front_right": (0.032, -0.027, -math.pi / 4, 0.50),
    "rear_left": (-0.032, 0.027, 3 * math.pi / 4, 0.25),
    "rear_right": (-0.032, -0.027, -3 * math.pi / 4, 0.75),
}

# Geometry from hello.xml. The rest pose is a modestly bent, four-foot stance.
COXA_LENGTH = 0.036
FEMUR_VECTOR = (0.030, 0.040)  # 50 mm printed femur, projected into the leg plane
TIBIA_VECTOR = (0.05975, 0.0988)  # Mean measured STL toe tip from the knee pivot
FEMUR_LENGTH = math.hypot(*FEMUR_VECTOR)
TIBIA_LENGTH = math.hypot(*TIBIA_VECTOR)
FEMUR_BASE_ANGLE = math.atan2(FEMUR_VECTOR[0], FEMUR_VECTOR[1])
TIBIA_BASE_ANGLE = math.atan2(TIBIA_VECTOR[0], TIBIA_VECTOR[1])
REST_FEMUR_ANGLE = -0.55
REST_TIBIA_ANGLE = 1.10
FOOT_RADIUS = 0.0015  # Small contact proxy aligned to each printed claw tip.


def leg_foot_geometry(femur_angle, tibia_angle):
    """Return radial reach and downward reach for a leg's current pitch angles."""
    total_femur_angle = femur_angle
    total_tibia_angle = femur_angle + tibia_angle
    femur_x = FEMUR_VECTOR[0] * math.cos(total_femur_angle) - FEMUR_VECTOR[1] * math.sin(total_femur_angle)
    femur_down = FEMUR_VECTOR[0] * math.sin(total_femur_angle) + FEMUR_VECTOR[1] * math.cos(total_femur_angle)
    tibia_x = TIBIA_VECTOR[0] * math.cos(total_tibia_angle) - TIBIA_VECTOR[1] * math.sin(total_tibia_angle)
    tibia_down = TIBIA_VECTOR[0] * math.sin(total_tibia_angle) + TIBIA_VECTOR[1] * math.cos(total_tibia_angle)
    return COXA_LENGTH + femur_x + tibia_x, femur_down + tibia_down


def rest_foot_geometry():
    """Return radial and downward reach for the default standing pose."""
    return leg_foot_geometry(REST_FEMUR_ANGLE, REST_TIBIA_ANGLE)


def smoothstep01(value):
    """Quintic easing with zero velocity and acceleration at both endpoints."""
    value = max(0.0, min(1.0, value))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


REST_RADIAL_REACH, REST_FOOT_DROP = rest_foot_geometry()
ROOT_STAND_HEIGHT = REST_FOOT_DROP + FOOT_RADIUS


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
            "schema_version": 2,
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


def inverse_kinematics(x_from_coxa, y_from_coxa, foot_down):
    """Solve coxa yaw and two pitch angles for a foot point in a leg's frame."""
    coxa_yaw = math.atan2(y_from_coxa, x_from_coxa)
    radial_distance = math.hypot(x_from_coxa, y_from_coxa)
    planar_x = radial_distance - COXA_LENGTH
    distance = math.hypot(planar_x, foot_down)

    minimum_reach = abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-5
    maximum_reach = FEMUR_LENGTH + TIBIA_LENGTH - 1e-5
    distance = min(max(distance, minimum_reach), maximum_reach)

    cos_delta = (
        distance * distance - FEMUR_LENGTH * FEMUR_LENGTH - TIBIA_LENGTH * TIBIA_LENGTH
    ) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
    # Choose the folded-knee branch: femur reaches outward and tibia points
    # back down toward the floor, which gives the splayed spider stance.
    delta = -math.acos(max(-1.0, min(1.0, cos_delta)))
    target_angle = math.atan2(planar_x, foot_down)
    inner_angle = math.atan2(
        TIBIA_LENGTH * math.sin(delta),
        FEMUR_LENGTH + TIBIA_LENGTH * math.cos(delta),
    )
    femur_angle = FEMUR_BASE_ANGLE - (target_angle - inner_angle)
    tibia_angle = (TIBIA_BASE_ANGLE - FEMUR_BASE_ANGLE) - delta
    return coxa_yaw, femur_angle, tibia_angle


def crawl_support_compensation(cycle, base_pose):
    """Shift the body target toward each three-foot support triangle smoothly."""
    legs = []
    for leg_name, (hip_x, hip_y, heading, phase_offset) in LEG_INFO.items():
        base_yaw = (base_pose or {}).get(f"{leg_name}_hip_abduction", 0.0)
        base_femur = (base_pose or {}).get(f"{leg_name}_hip_pitch", REST_FEMUR_ANGLE)
        base_tibia = (base_pose or {}).get(f"{leg_name}_knee_pitch", REST_TIBIA_ANGLE)
        reach, _ = leg_foot_geometry(base_femur, base_tibia)
        world_heading = heading + base_yaw
        nominal_foot = (
            hip_x + reach * math.cos(world_heading),
            hip_y + reach * math.sin(world_heading),
        )
        legs.append((leg_name, nominal_foot, phase_offset))

    legs.sort(key=lambda item: (item[2] + SWING_FRACTION / 2.0) % 1.0)
    supports = {}
    for leg_name, _, _ in legs:
        points = [point for other_name, point, _ in legs if other_name != leg_name]
        supports[leg_name] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    centers = [((offset + SWING_FRACTION / 2.0) % 1.0, name) for name, _, offset in legs]
    centers.sort()
    phase = cycle % 1.0
    for index, (left_center, left_leg) in enumerate(centers):
        next_center, next_leg = centers[(index + 1) % len(centers)]
        phase_value = phase
        if index == len(centers) - 1:
            next_center += 1.0
            if phase_value < left_center:
                phase_value += 1.0
        elif phase_value < left_center:
            continue
        if left_center <= phase_value <= next_center:
            blend = smoothstep01((phase_value - left_center) / (next_center - left_center))
            center_x = supports[left_leg][0] + blend * (supports[next_leg][0] - supports[left_leg][0])
            center_y = supports[left_leg][1] + blend * (supports[next_leg][1] - supports[left_leg][1])
            return -SUPPORT_SHIFT_FRACTION * center_x, -SUPPORT_SHIFT_FRACTION * center_y
    return 0.0, 0.0


def crawl_swinging_legs(gait_time):
    """Return the legs currently in swing for the sequential crawl phase."""
    cycle = gait_time * GAIT_FREQUENCY_HZ
    return {
        leg_name
        for leg_name, (_, _, _, phase_offset) in LEG_INFO.items()
        if (cycle + phase_offset) % 1.0 < SWING_FRACTION
    }


def crawl_targets(gait_time, mode, base_pose=None, blend=1.0, stance_corrections=None):
    """Generate crawl targets around a captured manual pose, then solve 3-DoF IK."""
    targets = {}
    if mode == "stop":
        for _, name in JOINTS:
            targets[name] = REST_FEMUR_ANGLE if name.endswith("hip_pitch") else (
                REST_TIBIA_ANGLE if name.endswith("knee_pitch") else 0.0
            )
        return targets

    blend = max(0.0, min(1.0, blend))

    direction = -1.0 if mode == "backward" else 1.0
    # During stance the stride offset moves from +half-stride to -half-stride,
    # so the target tangent is opposite the requested chassis rotation.
    turn_sign = -1.0 if mode == "left" else 1.0
    stride_metres = TURN_STRIDE_METRES if mode in ("left", "right") else STRIDE_METRES
    # The gait clock starts when a direction is pressed, not when MuJoCo starts.
    # That makes every held command begin at the same crawl phase, even after
    # a long idle or after switching direction.
    cycle = gait_time * GAIT_FREQUENCY_HZ
    support_comp_x, support_comp_y = crawl_support_compensation(cycle, base_pose)

    for leg_name, (hip_x, hip_y, heading, phase_offset) in LEG_INFO.items():
        base_yaw = (base_pose or {}).get(f"{leg_name}_hip_abduction", 0.0)
        base_femur = (base_pose or {}).get(f"{leg_name}_hip_pitch", REST_FEMUR_ANGLE)
        base_tibia = (base_pose or {}).get(f"{leg_name}_knee_pitch", REST_TIBIA_ANGLE)
        base_reach, base_foot_down = leg_foot_geometry(base_femur, base_tibia)
        base_heading = heading + base_yaw
        phase = (cycle + phase_offset) % 1.0
        if phase < SWING_FRACTION:
            swing_progress = smoothstep01(phase / SWING_FRACTION)
            # Move the foot forward only while it is clear of the floor. The
            # eased endpoints avoid a sharp reversal when swing meets stance.
            stride_offset = (-0.5 + swing_progress) * stride_metres
            foot_lift = FOOT_LIFT_METRES * math.sin(math.pi * swing_progress)
        else:
            stance_progress = smoothstep01(
                (phase - SWING_FRACTION) / (1.0 - SWING_FRACTION)
            )
            # Sweep the planted foot backward at a smooth rate to drive the
            # chassis forward through ground contact.
            stride_offset = (0.5 - stance_progress) * stride_metres
            foot_lift = 0.0

        if mode in ("forward", "backward"):
            travel_x = direction
            travel_y = 0.0
        else:
            # The ground-relative stance sweep produces the requested chassis yaw.
            travel_x = turn_sign * math.sin(base_heading)
            travel_y = -turn_sign * math.cos(base_heading)

        foot_world_x = (
            hip_x + base_reach * math.cos(base_heading)
            + stride_offset * travel_x + support_comp_x
        )
        foot_world_y = (
            hip_y + base_reach * math.sin(base_heading)
            + stride_offset * travel_y + support_comp_y
        )
        relative_x = foot_world_x - hip_x
        relative_y = foot_world_y - hip_y
        local_x = math.cos(heading) * relative_x + math.sin(heading) * relative_y
        local_y = -math.sin(heading) * relative_x + math.cos(heading) * relative_y
        stance_correction = (stance_corrections or {}).get(leg_name, 0.0)
        if phase >= SWING_FRACTION and stance_progress > 0.85:
            release_progress = smoothstep01((stance_progress - 0.85) / 0.15)
            stance_correction *= 1.0 - release_progress
        foot_down = (
            base_foot_down + STANCE_PRELOAD_METRES + stance_correction - foot_lift
        )
        yaw, femur, tibia = inverse_kinematics(local_x, local_y, foot_down)

        base_yaw = (base_pose or {}).get(f"{leg_name}_hip_abduction", 0.0)
        base_femur = (base_pose or {}).get(f"{leg_name}_hip_pitch", REST_FEMUR_ANGLE)
        base_tibia = (base_pose or {}).get(f"{leg_name}_knee_pitch", REST_TIBIA_ANGLE)
        targets[f"{leg_name}_hip_abduction"] = base_yaw + blend * (yaw - base_yaw)
        targets[f"{leg_name}_hip_pitch"] = base_femur + blend * (femur - base_femur)
        targets[f"{leg_name}_knee_pitch"] = base_tibia + blend * (tibia - base_tibia)

    return targets


def foot_contact_flags(data, foot_geom_ids, floor_geom_id):
    """Return whether each named foot geom currently contacts the floor geom."""
    contacts = {leg_name: False for leg_name in foot_geom_ids}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        for leg_name, foot_geom_id in foot_geom_ids.items():
            if {contact.geom1, contact.geom2} == {foot_geom_id, floor_geom_id}:
                contacts[leg_name] = True
    return contacts


def yaw_only_quaternion(quaternion):
    """Keep the current heading while asking the balance assist to level roll/pitch."""
    w, x, y, z = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def apply_pose_balance_assist(data, body_id, target_position, target_quaternion):
    """Apply a bounded virtual spring-damper to the free chassis in pose mode."""
    force = [
        BALANCE_POSITION_KP[axis] * (target_position[axis] - data.qpos[axis])
        - BALANCE_POSITION_KD[axis] * data.qvel[axis]
        for axis in range(3)
    ]
    force = [
        max(-BALANCE_FORCE_LIMIT_NEWTONS, min(BALANCE_FORCE_LIMIT_NEWTONS, component))
        for component in force
    ]

    current = tuple(float(value) for value in data.qpos[3:7])
    target_inverse = (target_quaternion[0], -target_quaternion[1], -target_quaternion[2], -target_quaternion[3])
    w, x, y, z = current
    tw, tx, ty, tz = target_inverse
    error_quaternion = (
        w * tw - x * tx - y * ty - z * tz,
        w * tx + x * tw + y * tz - z * ty,
        w * ty - x * tz + y * tw + z * tx,
        w * tz + x * ty - y * tx + z * tw,
    )
    if error_quaternion[0] < 0.0:
        error_quaternion = tuple(-value for value in error_quaternion)
    vector_norm = math.sqrt(sum(value * value for value in error_quaternion[1:]))
    if vector_norm > 1e-9:
        angle = 2.0 * math.atan2(vector_norm, max(0.0, error_quaternion[0]))
        rotation_error = [-angle * value / vector_norm for value in error_quaternion[1:]]
    else:
        rotation_error = [0.0, 0.0, 0.0]
    torque = [
        BALANCE_ROTATION_KP * rotation_error[axis] - BALANCE_ROTATION_KD * data.qvel[axis + 3]
        for axis in range(3)
    ]
    torque = [
        max(-BALANCE_TORQUE_LIMIT_NM, min(BALANCE_TORQUE_LIMIT_NM, component))
        for component in torque
    ]
    data.xfrc_applied[body_id, :3] = force
    data.xfrc_applied[body_id, 3:] = torque


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_names = [name for _, name in JOINTS]
    actuator_ids = {}
    joint_ids = {}
    for label, joint_name in JOINTS:
        actuator_name = joint_name + "_motor"
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if actuator_id < 0 or joint_id < 0:
            raise RuntimeError(f"Missing model joint or actuator for {label}")
        actuator_ids[joint_name] = actuator_id
        joint_ids[joint_name] = joint_id
    foot_geom_ids = {
        leg_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg_name}_foot")
        for leg_name in LEG_INFO
    }
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Start in a bent four-foot stance rather than with fully extended legs.
    data.qpos[2] = ROOT_STAND_HEIGHT
    rest_pose = {}
    for _, joint_name in JOINTS:
        angle = REST_FEMUR_ANGLE if joint_name.endswith("hip_pitch") else (
            REST_TIBIA_ANGLE if joint_name.endswith("knee_pitch") else 0.0
        )
        qpos_address = model.jnt_qposadr[joint_ids[joint_name]]
        data.qpos[qpos_address] = angle
        rest_pose[joint_name] = angle
    mujoco.mj_forward(model, data)
    last_foot_contacts = foot_contact_flags(data, foot_geom_ids, floor_geom_id)
    stance_corrections = {leg_name: 0.0 for leg_name in LEG_INFO}
    chassis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    balance_target = {
        "position": tuple(float(value) for value in data.qpos[:3]),
        "quaternion": yaw_only_quaternion(data.qpos[3:7]),
    }

    recorder = EpisodeRecorder(joint_names, LEG_INFO)
    root = tk.Tk()
    root.title("Midori | Quadruped control")
    root.geometry("500x900")
    root.minsize(420, 640)

    ttk.Label(root, text="MIDORI | QUADRUPED CONTROL", font=("Segoe UI", 14, "bold")).pack(
        padx=12, pady=(12, 4), anchor="w"
    )
    ttk.Label(
        root,
        text="Hold a direction button or arrow key to move. Release it to stop.",
        wraplength=460,
    ).pack(padx=12, pady=(0, 6), anchor="w")

    movement_mode = {
        "value": "stop",
        "started_at": data.time,
        "blend_start": None,
        "stop_started_at": None,
        "stop_from_pose": None,
        "last_targets": None,
        "prepare_started_at": None,
        "prepare_from_pose": None,
    }
    gait_base_pose = {}
    balance_assist_var = tk.BooleanVar(value=False)
    status = ttk.Label(
        root,
        text="Stopped in the standing pose.",
        wraplength=460,
    )
    status.pack(padx=12, pady=(0, 8), anchor="w")

    def start_gait(direction):
        previous_mode = movement_mode["value"]
        movement_mode["value"] = direction
        if previous_mode != direction:
            if previous_mode == "stop":
                movement_mode["prepare_started_at"] = data.time
                movement_mode["prepare_from_pose"] = {
                    name: float(data.qpos[model.jnt_qposadr[joint_ids[name]]])
                    for name in joint_names
                }
                movement_mode["started_at"] = (
                    data.time + WALK_STANCE_TRANSITION_SECONDS
                )
                movement_mode["blend_start"] = movement_mode["started_at"]
                movement_mode["stop_started_at"] = None
                movement_mode["stop_from_pose"] = None
                gait_base_pose.clear()
                gait_base_pose.update({
                    name: movement_mode["prepare_from_pose"][name]
                    for name in joint_names
                })
                for name in joint_names:
                    if name.endswith("hip_pitch"):
                        gait_base_pose[name] = WALK_STANCE_FEMUR_ANGLE
                    elif name.endswith("knee_pitch"):
                        gait_base_pose[name] = WALK_STANCE_TIBIA_ANGLE
                if recorder.active:
                    recorder.metadata["manual_pose_baseline_deg"] = {
                        name: math.degrees(value)
                        for name, value in movement_mode["prepare_from_pose"].items()
                    }
                    recorder.metadata["walk_stance_deg"] = {
                        "femur_pitch": math.degrees(WALK_STANCE_FEMUR_ANGLE),
                        "tibia_pitch": math.degrees(WALK_STANCE_TIBIA_ANGLE),
                    }
                    recorder._write_metadata()
            else:
                if movement_mode["prepare_started_at"] is None:
                    movement_mode["started_at"] = data.time
                    movement_mode["blend_start"] = None
            recorder.add_command_event(data.time, "command_start", direction)
        status.configure(
            text=f"Moving {direction}; easing into the crawl stance. Release to return to your slider pose."
        )

    def stop_gait():
        previous_mode = movement_mode["value"]
        movement_mode["value"] = "stop"
        if previous_mode != "stop":
            recorder.add_command_event(data.time, "command_release", "stop")
            balance_target["position"] = tuple(float(value) for value in data.qpos[:3])
            balance_target["quaternion"] = yaw_only_quaternion(data.qpos[3:7])
            movement_mode["stop_started_at"] = data.time
            movement_mode["stop_from_pose"] = dict(movement_mode["last_targets"] or {})
            movement_mode["blend_start"] = None
            movement_mode["prepare_started_at"] = None
            movement_mode["prepare_from_pose"] = None
            status.configure(text="Stopping smoothly at your manual slider pose.")
        elif movement_mode["stop_started_at"] is None:
            status.configure(text="Stopped; holding the current manual joint pose.")

    def toggle_balance_assist():
        balance_target["position"] = tuple(float(value) for value in data.qpos[:3])
        balance_target["quaternion"] = yaw_only_quaternion(data.qpos[3:7])
        enabled = bool(balance_assist_var.get())
        if recorder.active:
            recorder.metadata["manual_pose_balance_assist_enabled"] = enabled
            recorder.add_command_event(
                data.time,
                "balance_assist_enabled" if enabled else "balance_assist_disabled",
                movement_mode["value"],
            )
            recorder._write_metadata()

    drive_frame = ttk.LabelFrame(root, text="Hold to move")
    drive_frame.pack(fill="x", padx=12, pady=(0, 8))
    drive_buttons = [
        ("Forward", "forward"),
        ("Backward", "backward"),
        ("Turn Left", "left"),
        ("Turn Right", "right"),
    ]
    for index, (label, mode) in enumerate(drive_buttons):
        button = ttk.Button(drive_frame, text=label)
        button.grid(row=index // 2, column=index % 2, sticky="ew", padx=4, pady=4)
        button.bind("<ButtonPress-1>", lambda _event, value=mode: start_gait(value))
        button.bind("<ButtonRelease-1>", lambda _event: stop_gait())
    ttk.Button(drive_frame, text="Stop", command=stop_gait).grid(
        row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4
    )
    ttk.Checkbutton(
        drive_frame,
        text="Optional virtual pose stabilization (off by default; disables natural falling)",
        variable=balance_assist_var,
        command=toggle_balance_assist,
    ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=4)
    drive_frame.columnconfigure(0, weight=1)
    drive_frame.columnconfigure(1, weight=1)

    capture_frame = ttk.LabelFrame(root, text="Dataset episode capture")
    capture_frame.pack(fill="x", padx=12, pady=(0, 8))
    ttk.Label(capture_frame, text="Task:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
    task_var = tk.StringVar(value="walk_forward")
    task_box = ttk.Combobox(
        capture_frame,
        textvariable=task_var,
        values=(
            "stand_balance", "walk_forward", "walk_backward",
            "turn_left", "turn_right", "mixed_command_sequence",
        ),
        state="readonly",
        width=25,
    )
    task_box.grid(row=0, column=1, columnspan=2, padx=5, pady=4, sticky="ew")
    outcome_var = tk.StringVar(value="incomplete")
    ttk.Label(capture_frame, text="Outcome:").grid(row=1, column=0, padx=5, pady=4, sticky="w")
    outcome_box = ttk.Combobox(
        capture_frame,
        textvariable=outcome_var,
        values=("success", "failure_fall", "failure_collision", "timeout", "operator_stop", "incomplete"),
        state="readonly",
        width=25,
    )
    outcome_box.grid(row=1, column=1, columnspan=2, padx=5, pady=4, sticky="ew")
    capture_status = ttk.Label(
        capture_frame,
        text="Not recording. Start one episode before demonstrating a task.",
        wraplength=460,
    )
    capture_status.grid(row=2, column=0, columnspan=3, padx=5, pady=4, sticky="w")
    sample_clock = {"next_sim_time": None}

    def start_episode():
        try:
            episode_id = recorder.start(
                task_var.get(), data.time, model, balance_assist_var.get()
            )
        except (OSError, RuntimeError) as error:
            capture_status.configure(text=f"Could not start episode: {error}")
            return
        sample_clock["next_sim_time"] = data.time + DATASET_SAMPLE_PERIOD_SECONDS
        capture_status.configure(
            text=f"Recording {episode_id} at 50 Hz simulation time. Demonstrate the selected task, then end it."
        )
        start_episode_button.configure(state="disabled")
        end_episode_button.configure(state="normal")
        task_box.configure(state="disabled")

    def end_episode():
        episode_dir = recorder.finish(data.time, outcome_var.get())
        if episode_dir is None:
            return
        capture_status.configure(
            text=f"Saved {recorder.row_count} rows to {episode_dir.relative_to(HERE.parent)}"
        )
        start_episode_button.configure(state="normal")
        end_episode_button.configure(state="disabled")
        task_box.configure(state="readonly")

    start_episode_button = ttk.Button(capture_frame, text="Start episode", command=start_episode)
    start_episode_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
    end_episode_button = ttk.Button(
        capture_frame, text="End episode and save", command=end_episode, state="disabled"
    )
    end_episode_button.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
    capture_frame.columnconfigure(1, weight=1)
    capture_frame.columnconfigure(2, weight=1)

    key_modes = {
        "w": "forward", "up": "forward",
        "s": "backward", "down": "backward",
        "a": "left", "left": "left",
        "d": "right", "right": "right",
    }
    pressed_keys = []

    def key_down(event):
        key = event.keysym.lower()
        mode = key_modes.get(key)
        if mode is None:
            return
        if key not in pressed_keys:
            pressed_keys.append(key)
        start_gait(mode)
        return "break"

    def key_up(event):
        key = event.keysym.lower()
        if key not in key_modes:
            return
        if key in pressed_keys:
            pressed_keys.remove(key)
        if pressed_keys:
            start_gait(key_modes[pressed_keys[-1]])
        else:
            stop_gait()
        return "break"

    root.bind_all("<KeyPress>", key_down)
    root.bind_all("<KeyRelease>", key_up)

    controls_area = ttk.Frame(root)
    controls_area.pack(fill="both", expand=True, padx=6)
    canvas = tk.Canvas(controls_area, highlightthickness=0)
    scrollbar = ttk.Scrollbar(controls_area, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    controls = ttk.Frame(canvas)
    controls_window = canvas.create_window((0, 0), window=controls, anchor="nw")
    sliders = {}

    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def fit_controls(event):
        canvas.itemconfigure(controls_window, width=event.width)
        for slider in sliders.values():
            slider.configure(length=max(220, event.width - 40))

    controls.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", fit_controls)

    def scroll_controls(event):
        if event.delta:
            canvas.yview_scroll(-int(event.delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")

    root.bind_all("<MouseWheel>", scroll_controls)
    root.bind_all("<Button-4>", scroll_controls)
    root.bind_all("<Button-5>", scroll_controls)

    for label, joint_name in JOINTS:
        joint_limits = model.jnt_range[joint_ids[joint_name]]
        slider_min = math.floor(math.degrees(joint_limits[0]))
        slider_max = math.ceil(math.degrees(joint_limits[1]))
        row = ttk.Frame(controls)
        row.pack(fill="x", padx=12, pady=3)
        value_label = ttk.Label(row, text=f"{label}: {math.degrees(rest_pose[joint_name]):.0f} deg")
        value_label.pack(anchor="w")
        slider = tk.Scale(
            row, from_=slider_min, to=slider_max, resolution=1, orient="horizontal",
            showvalue=False, length=300,
        )
        slider.set(math.degrees(rest_pose[joint_name]))
        slider.pack(fill="x")
        slider.configure(
            command=lambda value, widget=value_label, title=label:
                widget.configure(text=f"{title}: {float(value):.0f} deg")
        )
        sliders[joint_name] = slider

    def reset_pose():
        stop_gait()
        for joint_name, slider in sliders.items():
            slider.set(math.degrees(rest_pose[joint_name]))

    ttk.Button(root, text="Stop and return to standing pose", command=reset_pose).pack(
        padx=12, pady=10, fill="x", side="bottom"
    )

    def advance_simulation_step():
        nonlocal last_foot_contacts
        mode = movement_mode["value"]
        if mode == "stop":
            stance_corrections.update({leg_name: 0.0 for leg_name in LEG_INFO})
            slider_targets = {
                joint_name: math.radians(float(sliders[joint_name].get()))
                for joint_name in joint_names
            }
            stop_started_at = movement_mode["stop_started_at"]
            if stop_started_at is not None:
                progress = min(1.0, max(0.0, data.time - stop_started_at) / 0.25)
                blend = smoothstep01(progress)
                start_targets = movement_mode["stop_from_pose"] or slider_targets
                targets = {
                    joint_name: start_targets.get(joint_name, slider_targets[joint_name])
                    + blend * (
                        slider_targets[joint_name]
                        - start_targets.get(joint_name, slider_targets[joint_name])
                    )
                    for joint_name in joint_names
                }
                if progress >= 1.0:
                    movement_mode["stop_started_at"] = None
                    movement_mode["stop_from_pose"] = None
                    status.configure(text="Stopped; holding the current manual joint pose.")
            else:
                targets = slider_targets
        else:
            prepare_start = movement_mode["prepare_started_at"]
            if prepare_start is not None:
                prepare_time = max(0.0, data.time - prepare_start)
                progress = min(1.0, prepare_time / WALK_STANCE_TRANSITION_SECONDS)
                blend = smoothstep01(progress)
                start_pose = movement_mode["prepare_from_pose"] or gait_base_pose
                targets = {
                    name: start_pose[name]
                    + blend * (gait_base_pose[name] - start_pose[name])
                    for name in joint_names
                }
                if progress >= 1.0:
                    movement_mode["prepare_started_at"] = None
                    movement_mode["prepare_from_pose"] = None
            else:
                gait_time = max(0.0, data.time - movement_mode["started_at"])
                swinging_legs = crawl_swinging_legs(gait_time)
                for leg_name in LEG_INFO:
                    if leg_name in swinging_legs:
                        stance_corrections[leg_name] = 0.0
                    elif last_foot_contacts[leg_name]:
                        stance_corrections[leg_name] = max(
                            0.0,
                            stance_corrections[leg_name]
                            - STANCE_RELEASE_RATE_MPS * model.opt.timestep,
                        )
                    else:
                        stance_corrections[leg_name] = min(
                            STANCE_SEARCH_MAX_METRES,
                            stance_corrections[leg_name]
                            + STANCE_SEARCH_RATE_MPS * model.opt.timestep,
                        )
                blend = 1.0
                blend_start = movement_mode["blend_start"]
                if blend_start is not None:
                    blend_time = max(0.0, data.time - blend_start)
                    progress = min(1.0, blend_time / WALK_GAIT_BLEND_SECONDS)
                    blend = smoothstep01(progress)
                    if progress >= 1.0:
                        movement_mode["blend_start"] = None
                targets = crawl_targets(
                    gait_time, mode, gait_base_pose, blend, stance_corrections
                )
        movement_mode["last_targets"] = dict(targets)

        for joint_name, target in targets.items():
            data.ctrl[actuator_ids[joint_name]] = target

        data.xfrc_applied[:] = 0.0
        if balance_assist_var.get() and mode == "stop":
            apply_pose_balance_assist(
                data,
                chassis_body_id,
                balance_target["position"],
                balance_target["quaternion"],
            )

        mujoco.mj_step(model, data)
        last_foot_contacts = foot_contact_flags(data, foot_geom_ids, floor_geom_id)

        next_sample_time = sample_clock["next_sim_time"]
        if (
            recorder.active
            and next_sample_time is not None
            and data.time + 1e-12 >= next_sample_time
        ):
            row = {
                "episode_id": recorder.episode_id,
                "step_index": recorder.row_count,
                "episode_time_s": data.time - recorder.start_sim_time,
                "sim_time_s": data.time,
                "recorded_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "movement_mode": movement_mode["value"],
                "balance_assist_active": int(
                    balance_assist_var.get() and movement_mode["value"] == "stop"
                ),
                "root_x_m": data.qpos[0], "root_y_m": data.qpos[1], "root_z_m": data.qpos[2],
                "root_quat_w": data.qpos[3], "root_quat_x": data.qpos[4],
                "root_quat_y": data.qpos[5], "root_quat_z": data.qpos[6],
                "root_vx_mps": data.qvel[0], "root_vy_mps": data.qvel[1], "root_vz_mps": data.qvel[2],
                "root_wx_radps": data.qvel[3], "root_wy_radps": data.qvel[4], "root_wz_radps": data.qvel[5],
            }
            for joint_name in joint_names:
                joint_id = joint_ids[joint_name]
                qpos_address = model.jnt_qposadr[joint_id]
                dof_address = model.jnt_dofadr[joint_id]
                row[f"target_{joint_name}_deg"] = math.degrees(targets[joint_name])
                row[f"actual_{joint_name}_deg"] = math.degrees(data.qpos[qpos_address])
                row[f"actual_{joint_name}_vel_degps"] = math.degrees(data.qvel[dof_address])
            for leg_name in LEG_INFO:
                foot_position = data.geom_xpos[foot_geom_ids[leg_name]]
                for axis_index, axis_name in enumerate(("x", "y", "z")):
                    row[f"foot_{leg_name}_{axis_name}_m"] = foot_position[axis_index]
            for leg_name, in_contact in last_foot_contacts.items():
                row[f"foot_{leg_name}_contact"] = int(in_contact)
            recorder.write(row)
            while sample_clock["next_sim_time"] <= data.time + 1e-12:
                sample_clock["next_sim_time"] += DATASET_SAMPLE_PERIOD_SECONDS

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -40
            viewer.cam.distance = 0.44
            viewer.cam.lookat[:] = [0.0, 0.0, ROOT_STAND_HEIGHT * 0.45]
            root.protocol("WM_DELETE_WINDOW", root.destroy)

            last_wall_time = time.perf_counter()
            simulation_accumulator = 0.0
            max_catchup_steps = max(1, int(0.10 / model.opt.timestep))
            while viewer.is_running():
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    break

                now = time.perf_counter()
                wall_elapsed = min(now - last_wall_time, 0.10)
                last_wall_time = now
                simulation_accumulator += wall_elapsed
                steps_this_frame = 0
                while (
                    simulation_accumulator + 1e-12 >= model.opt.timestep
                    and steps_this_frame < max_catchup_steps
                ):
                    advance_simulation_step()
                    simulation_accumulator -= model.opt.timestep
                    steps_this_frame += 1

                viewer.sync()
                time_until_next_step = model.opt.timestep - simulation_accumulator
                if time_until_next_step > 0.0:
                    time.sleep(min(time_until_next_step, 0.002))

            try:
                root.destroy()
            except tk.TclError:
                pass
    finally:
        recorder.finish(data.time, "interrupted")


if __name__ == "__main__":
    main()
