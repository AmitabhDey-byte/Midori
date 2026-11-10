"""Replay a trained Midori policy and its scripted teacher in MuJoCo."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mujoco"))
sys.path.insert(0, str(ROOT / "training"))
import control_test as gait
from physics_policy_10m import TemporalPhysicsPolicy


JOINT_NAMES = [name for _, name in gait.JOINTS]
COMMANDS = ["stop", "forward", "backward", "left", "right"]
LEGS = ["front_left", "front_right", "rear_left", "rear_right"]
SCREENSHOT_POSE_DEG = {
    "front_left_hip_abduction": -11.0,
    "front_right_hip_abduction": 19.0,
}
SCREENSHOT_FEMUR_DEG = -86.0
SCREENSHOT_TIBIA_DEG = 92.0


class MidoriLocomotionPolicy(nn.Module):
    def __init__(self, input_size=54, hidden_size=128, layers=2, action_size=12):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.10 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, action_size),
            nn.Tanh(),
        )

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1])


def load_checkpoint(path: Path):


    allow = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allow):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("gait_version") != gait.GAIT_CONTROLLER_VERSION:
        raise ValueError(
            f"Checkpoint gait {checkpoint.get('gait_version')!r} does not match "
            f"the installed simulator gait {gait.GAIT_CONTROLLER_VERSION!r}."
        )
    if (
        checkpoint.get("joints") != JOINT_NAMES
        or checkpoint.get("commands") != COMMANDS
        or checkpoint.get("feature_count") != 54
    ):
        raise ValueError("Checkpoint joint order or observation schema does not match this evaluator.")
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
        model = TemporalPhysicsPolicy(
            input_size=checkpoint["feature_count"],
            context=checkpoint["context_steps"],
            width=config["width"], heads=config["heads"],
            layers=config["layers"], ff=config["ff"],
            action_size=len(checkpoint["joints"]),
        )
    else:
        model = MidoriLocomotionPolicy(
            input_size=checkpoint["feature_count"],
            hidden_size=checkpoint["hidden_size"],
            layers=checkpoint["layers"],
            action_size=len(checkpoint["joints"]),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def quat_inverse_rotate(q, vector):
    w, x, y, z = q
    inverse_vector = np.asarray([-x, -y, -z], dtype=np.float32)
    cross = 2.0 * np.cross(inverse_vector, vector)
    return np.asarray(vector + w * cross + np.cross(inverse_vector, cross), dtype=np.float32)


def quaternion_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def observe(model, data, joint_ids, foot_geom_ids, command):
    q = np.asarray(data.qpos[3:7], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    angle_limits = np.asarray(model["angle_limits_deg"], dtype=np.float32)

    actual = np.asarray(
        [math.degrees(data.qpos[model["qpos_addresses"][name]]) for name in JOINT_NAMES],
        dtype=np.float32,
    )
    actual_norm = 2 * (actual - angle_limits[:, 0]) / (angle_limits[:, 1] - angle_limits[:, 0]) - 1
    velocity = np.asarray(
        [math.degrees(data.qvel[model["dof_addresses"][name]]) for name in JOINT_NAMES],
        dtype=np.float32,
    ) / 360.0

    root_position = np.asarray(data.qpos[:3], dtype=np.float32)
    foot_local = []
    contacts = gait.foot_contact_flags(data, foot_geom_ids, model["floor_geom_id"])
    for leg in LEGS:
        foot_world = np.asarray(data.geom_xpos[foot_geom_ids[leg]], dtype=np.float32)
        foot_local.extend(quat_inverse_rotate(q, foot_world - root_position))

    command_onehot = np.eye(len(COMMANDS), dtype=np.float32)[COMMANDS.index(command)]
    body = np.asarray(
        [
            data.qpos[2], roll, pitch,
            data.qvel[0], data.qvel[1], data.qvel[2],
            data.qvel[3], data.qvel[4], data.qvel[5],
        ],
        dtype=np.float32,
    )
    contact_values = np.asarray([contacts[leg] for leg in LEGS], dtype=np.float32)
    observation = np.concatenate(
        [command_onehot, actual_norm, velocity, body, np.asarray(foot_local, dtype=np.float32), contact_values]
    )
    if observation.shape != (54,):
        raise RuntimeError(f"Expected a 54-value observation, got {observation.shape}.")
    normalized = (observation - model["input_mean"]) / model["input_std"]
    return normalized.astype(np.float32), contacts, roll, pitch


def make_simulation(xml_path: Path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    }
    actuator_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
        for name in JOINT_NAMES
    }
    foot_geom_ids = {
        leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
        for leg in LEGS
    }
    if any(index < 0 for index in [*joint_ids.values(), *actuator_ids.values(), *foot_geom_ids.values()]):
        raise RuntimeError("The MuJoCo model is missing a required joint, actuator, or foot geom.")
    qpos_addresses = {name: int(model.jnt_qposadr[jid]) for name, jid in joint_ids.items()}
    dof_addresses = {name: int(model.jnt_dofadr[jid]) for name, jid in joint_ids.items()}
    return model, data, joint_ids, actuator_ids, foot_geom_ids, qpos_addresses, dof_addresses


def start_from_screenshot(xml_path, duration, settle_seconds, policy, checkpoint):
    mj_model, data, joint_ids, actuator_ids, foot_geom_ids, qpos_addresses, dof_addresses = make_simulation(xml_path)
    data.qpos[2] = gait.ROOT_STAND_HEIGHT
    manual_pose = {}
    for name in JOINT_NAMES:
        if name.endswith("hip_abduction"):
            degrees = SCREENSHOT_POSE_DEG.get(name, 0.0)
        elif name.endswith("hip_pitch"):
            degrees = SCREENSHOT_FEMUR_DEG
        else:
            degrees = SCREENSHOT_TIBIA_DEG
        jid = joint_ids[name]
        lower, upper = mj_model.jnt_range[jid]
        angle = max(float(lower), min(float(upper), math.radians(degrees)))
        manual_pose[name] = angle
        data.qpos[qpos_addresses[name]] = angle
        data.ctrl[actuator_ids[name]] = angle
    mujoco.mj_forward(mj_model, data)


    for _ in range(round(settle_seconds / mj_model.opt.timestep)):
        for name in JOINT_NAMES:
            data.ctrl[actuator_ids[name]] = manual_pose[name]
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(mj_model, data)

    source_pose = {name: float(data.qpos[qpos_addresses[name]]) for name in JOINT_NAMES}
    walk_pose = dict(source_pose)
    for name in JOINT_NAMES:
        if name.endswith("hip_pitch"):
            walk_pose[name] = gait.WALK_STANCE_FEMUR_ANGLE
        elif name.endswith("knee_pitch"):
            walk_pose[name] = gait.WALK_STANCE_TIBIA_ANGLE

    sim_context = {
        "model": mj_model,
        "data": data,
        "joint_ids": joint_ids,
        "actuator_ids": actuator_ids,
        "foot_geom_ids": foot_geom_ids,
        "qpos_addresses": qpos_addresses,
        "dof_addresses": dof_addresses,
        "floor_geom_id": mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
        "source_pose": source_pose,
        "walk_pose": walk_pose,
        "duration": duration,
        "checkpoint": checkpoint,
        "policy": policy,
        "angle_limits_deg": checkpoint["angle_limits_deg"],
        "input_mean": checkpoint["input_mean"],
        "input_std": checkpoint["input_std"],
    }


    history = deque(maxlen=int(checkpoint["context_steps"]))
    substeps_per_observation = round(1.0 / (checkpoint["sample_rate_hz"] * mj_model.opt.timestep))
    for _ in range(int(checkpoint["context_steps"])):
        obs, _, _, _ = observe(sim_context, data, joint_ids, foot_geom_ids, "stop")
        history.append(obs)
        for _ in range(substeps_per_observation):
            for name in JOINT_NAMES:
                data.ctrl[actuator_ids[name]] = manual_pose[name]
            mujoco.mj_step(mj_model, data)
    sim_context["history"] = history
    return sim_context


def angles_from_policy(context, observation):
    checkpoint = context["checkpoint"]
    tensor = torch.from_numpy(np.stack(context["history"])[None, ...])
    with torch.no_grad():
        output = context["policy"](tensor)


        action = output[0] if isinstance(output, tuple) else output
        normalized_target = action.cpu().numpy()[0]
    limits = np.asarray(checkpoint["angle_limits_deg"], dtype=np.float32)
    target_degrees = limits[:, 0] + 0.5 * (normalized_target + 1.0) * (limits[:, 1] - limits[:, 0])
    return {name: math.radians(float(angle)) for name, angle in zip(JOINT_NAMES, target_degrees)}


def run_rollout(context, controller_name, viewer=None, command="forward"):
    model = context["model"]
    data = context["data"]
    joint_ids = context["joint_ids"]
    actuator_ids = context["actuator_ids"]
    foot_geom_ids = context["foot_geom_ids"]
    source_pose = context["source_pose"]
    walk_pose = context["walk_pose"]
    checkpoint = context["checkpoint"]
    policy = context["policy"]
    history = context["history"]
    corrections = {leg: 0.0 for leg in LEGS}
    last_contacts = gait.foot_contact_flags(data, foot_geom_ids, context["floor_geom_id"])
    dt = float(model.opt.timestep)
    observation_steps = max(1, round(1.0 / (checkpoint["sample_rate_hz"] * dt)))
    training_stride = int(checkpoint.get("sample_stride", 1 if "model_config" in checkpoint else 2))
    action_steps = observation_steps * training_stride
    total_steps = round(context["duration"] / dt)
    initial_position = np.asarray(data.qpos[:3], dtype=np.float64).copy()
    initial_yaw = quaternion_yaw(data.qpos[3:7])
    previous_yaw = initial_yaw
    accumulated_yaw = 0.0
    min_root_z = float(data.qpos[2])
    max_roll = 0.0
    max_pitch = 0.0
    contact_counts = [0, 0, 0, 0, 0]
    foot_contact_samples = {leg: 0 for leg in LEGS}
    root_z_samples = []
    target_limit_violations = 0
    max_actual_limit_violation = 0.0
    teacher_forced_action_errors_deg = []
    rows = 0
    last_targets = dict(source_pose)
    viewer_frame_start = time.perf_counter()

    for step in range(total_steps):
        elapsed = step * dt
        if controller_name in ("scripted", "teacher_forced"):
            if elapsed < gait.WALK_STANCE_TRANSITION_SECONDS:
                blend = gait.smoothstep01(elapsed / gait.WALK_STANCE_TRANSITION_SECONDS)
                targets = {
                    name: source_pose[name] + blend * (walk_pose[name] - source_pose[name])
                    for name in JOINT_NAMES
                }
            else:
                gait_time = elapsed - gait.WALK_STANCE_TRANSITION_SECONDS
                swinging = gait.crawl_swinging_legs(gait_time)
                for leg in LEGS:
                    if leg in swinging:
                        corrections[leg] = 0.0
                    elif last_contacts[leg]:
                        corrections[leg] = max(
                            0.0, corrections[leg] - gait.STANCE_RELEASE_RATE_MPS * dt
                        )
                    else:
                        corrections[leg] = min(
                            gait.STANCE_SEARCH_MAX_METRES,
                            corrections[leg] + gait.STANCE_SEARCH_RATE_MPS * dt,
                        )
                crawl_blend = gait.smoothstep01(
                    min(1.0, gait_time / gait.WALK_GAIT_BLEND_SECONDS)
                )
                targets = gait.crawl_targets(
                    gait_time, "forward", walk_pose, crawl_blend, corrections
                )
        else:
            observation = None
        if step % observation_steps == 0:
            observation, _, _, _ = observe(context, data, joint_ids, foot_geom_ids, command)
            history.append(observation)
            if controller_name == "teacher_forced" and step % action_steps == 0:
                predicted = angles_from_policy(context, observation)
                teacher_forced_action_errors_deg.extend(
                    abs(
                        math.degrees(
                            predicted[name]
                            - max(
                                float(model.jnt_range[joint_ids[name], 0]),
                                min(float(model.jnt_range[joint_ids[name], 1]), targets[name]),
                            )
                        )
                    )
                    for name in JOINT_NAMES
                )
            if controller_name == "policy" and step % action_steps == 0:
                targets = angles_from_policy(context, observation)
                last_targets = targets

        use_teacher_targets = controller_name in ("scripted", "teacher_forced")
        applied_targets = targets if use_teacher_targets else last_targets
        for name, target in applied_targets.items():
            jid = joint_ids[name]
            lower, upper = model.jnt_range[jid]
            if target < lower - 1e-8 or target > upper + 1e-8:
                target_limit_violations += 1
            data.ctrl[actuator_ids[name]] = target

        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)
        last_contacts = gait.foot_contact_flags(data, foot_geom_ids, context["floor_geom_id"])
        min_root_z = min(min_root_z, float(data.qpos[2]))
        root_z_samples.append(float(data.qpos[2]))
        count = sum(last_contacts.values())
        contact_counts[count] += 1
        for leg in LEGS:
            foot_contact_samples[leg] += int(last_contacts[leg])
        q = data.qpos[3:7]
        w, x, y, z = q
        yaw = quaternion_yaw(q)
        accumulated_yaw += math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
        previous_yaw = yaw
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        max_roll = max(max_roll, abs(roll))
        max_pitch = max(max_pitch, abs(pitch))
        for name, jid in joint_ids.items():
            qpos = data.qpos[model.jnt_qposadr[jid]]
            lower, upper = model.jnt_range[jid]
            max_actual_limit_violation = max(
                max_actual_limit_violation, float(max(0.0, lower - qpos, qpos - upper))
            )
        rows += 1
        if viewer is not None and ((step + 1) % observation_steps == 0 or step == total_steps - 1):
            viewer.sync()
            frame_seconds = observation_steps * dt
            remaining = frame_seconds - (time.perf_counter() - viewer_frame_start)
            if remaining > 0.0:
                time.sleep(remaining)
            viewer_frame_start = time.perf_counter()

    final_position = np.asarray(data.qpos[:3], dtype=np.float64)
    forward_displacement = float(final_position[0] - initial_position[0])
    return {
        "controller": controller_name,
        "command": command,
        "duration_sim_s": context["duration"],
        "gravity_mps2": [float(v) for v in model.opt.gravity],
        "virtual_stabilization": False,
        "initial_root_xyz_m": initial_position.tolist(),
        "final_root_xyz_m": final_position.tolist(),
        "net_yaw_change_deg": math.degrees(accumulated_yaw),
        "forward_displacement_m": forward_displacement,
        "mean_forward_speed_mps": forward_displacement / context["duration"],
        "min_root_z_m": min_root_z,
        "max_abs_roll_deg": math.degrees(max_roll),
        "max_abs_pitch_deg": math.degrees(max_pitch),
        "root_z_min_max_m": [min(root_z_samples), max(root_z_samples)],
        "foot_contact_frame_fraction_by_count": {
            str(count): fraction / rows for count, fraction in enumerate(contact_counts)
        },
        "per_foot_contact_fraction": {
            leg: count / rows for leg, count in foot_contact_samples.items()
        },
        "target_joint_limit_violation_count": target_limit_violations,
        "max_actual_joint_limit_violation_deg": math.degrees(max_actual_limit_violation),
        "teacher_forced_mean_target_error_deg": (
            float(np.mean(teacher_forced_action_errors_deg))
            if teacher_forced_action_errors_deg else None
        ),
        "fell_by_simple_threshold": bool(
            min_root_z < 0.045 or max(max_roll, max_pitch) > math.radians(35.0)
        ),
        "sample_count_at_50hz": round(context["duration"] * checkpoint["sample_rate_hz"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "training" / "best_policy_10m.pt")
    parser.add_argument("--xml", type=Path, default=ROOT / "mujoco" / "hello.xml")
    parser.add_argument("--command", choices=("forward", "backward", "left", "right"), default="forward",
                        help="Movement command supplied to the learned policy")
    parser.add_argument("--duration", type=float, default=8.0, help="Command duration in simulation seconds")
    parser.add_argument("--settle", type=float, default=3.0, help="Seconds to settle the screenshot pose under gravity")
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "policy_simulation_results.json")
    parser.add_argument("--viewer", action="store_true", help="Show the learned policy rollout in the MuJoCo viewer")
    args = parser.parse_args()

    policy, checkpoint = load_checkpoint(args.checkpoint)
    reports = {}
    if args.viewer:
        context = start_from_screenshot(args.xml, args.duration, args.settle, policy, checkpoint)
        with mujoco.viewer.launch_passive(context["model"], context["data"]) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -40
            viewer.cam.distance = 0.44
            viewer.cam.lookat[:] = [0.0, 0.0, gait.ROOT_STAND_HEIGHT * 0.45]
            viewer.sync()
            reports["policy"] = run_rollout(context, "policy", viewer, args.command)
            print("Rollout complete. Close the MuJoCo viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    else:
        names = ("policy", "teacher_forced", "scripted") if args.command == "forward" else ("policy",)
        for name in names:
            context = start_from_screenshot(args.xml, args.duration, args.settle, policy, checkpoint)
            reports[name] = run_rollout(context, name, command=args.command)

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "simulator_xml": str(args.xml.resolve()),
        "gait_version": checkpoint["gait_version"],
        "command": args.command,
        "initial_pose": "screenshot slider pose; angles clamped to model limits",
        "settle_seconds_with_gravity": args.settle,
        "comparison": reports,
        "interpretation": "Imitation metrics do not establish stable locomotion; inspect displacement, tilt, height, and toe-contact fractions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"Saved report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
