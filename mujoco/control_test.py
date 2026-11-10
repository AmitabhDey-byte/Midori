"""MuJoCo controls and viewer for the Midori quadruped."""

import math
import sys
import tkinter as tk
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer

from episode_recorder import EpisodeRecorder
from gait import (
    DATASET_SAMPLE_PERIOD_SECONDS,
    JOINTS,
    LEG_INFO,
    ROOT_STAND_HEIGHT,
    REST_FEMUR_ANGLE,
    REST_TIBIA_ANGLE,
    STANCE_RELEASE_RATE_MPS,
    STANCE_SEARCH_MAX_METRES,
    STANCE_SEARCH_RATE_MPS,
    WALK_GAIT_BLEND_SECONDS,
    WALK_STANCE_FEMUR_ANGLE,
    WALK_STANCE_TIBIA_ANGLE,
    WALK_STANCE_TRANSITION_SECONDS,
    crawl_swinging_legs,
    crawl_targets,
    foot_contact_flags,
    smoothstep01,
)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "hello.xml"
LOG_DIR = HERE / "logs"
BALANCE_POSITION_KP = (220.0, 220.0, 420.0)
BALANCE_POSITION_KD = (28.0, 28.0, 38.0)
BALANCE_ROTATION_KP = 5.0
BALANCE_ROTATION_KD = 0.9
BALANCE_FORCE_LIMIT_NEWTONS = 18.0
BALANCE_TORQUE_LIMIT_NM = 2.5

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


    policy_runtime = None
    policy_model = None
    policy_checkpoint = None
    policy_load_error = None
    training_dir = HERE.parent / "training"
    policy_path = training_dir / "best_policy_10m.pt"
    try:
        sys.path.insert(0, str(training_dir))
        import evaluate_policy as policy_runtime

        if policy_path.exists():
            policy_model, policy_checkpoint = policy_runtime.load_checkpoint(policy_path)
        else:
            policy_load_error = f"Policy checkpoint not found: {policy_path}"
    except Exception as error:
        policy_load_error = str(error)

    policy_history = deque(maxlen=int(policy_checkpoint["context_steps"])) if policy_checkpoint else deque()
    policy_state = {"next_inference_time": data.time, "targets": dict(rest_pose)}
    policy_context = None
    if policy_checkpoint:
        qpos_addresses = {name: int(model.jnt_qposadr[joint_ids[name]]) for name in joint_names}
        dof_addresses = {name: int(model.jnt_dofadr[joint_ids[name]]) for name in joint_names}
        policy_context = {
            "angle_limits_deg": policy_checkpoint["angle_limits_deg"],
            "input_mean": policy_checkpoint["input_mean"],
            "input_std": policy_checkpoint["input_std"],
            "qpos_addresses": qpos_addresses,
            "dof_addresses": dof_addresses,
            "floor_geom_id": floor_geom_id,
            "checkpoint": policy_checkpoint,
            "policy": policy_model,
            "history": policy_history,
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

    policy_frame = ttk.LabelFrame(root, text="Policy correction capture")
    policy_frame.pack(fill="x", padx=12, pady=(0, 8))
    policy_enabled_var = tk.BooleanVar(value=False)
    teacher_override_var = tk.BooleanVar(value=False)

    def seed_policy_history():
        if not policy_context:
            return
        policy_history.clear()
        for _ in range(int(policy_checkpoint["context_steps"])):
            observation, _, _, _ = policy_runtime.observe(
                policy_context, data, joint_ids, foot_geom_ids, "stop"
            )
            policy_history.append(observation)
        policy_state["targets"] = dict(rest_pose)
        policy_state["next_inference_time"] = data.time

    def toggle_policy_mode():
        enabled = bool(policy_enabled_var.get())
        if enabled:
            seed_policy_history()
        if recorder.active:
            recorder.add_command_event(
                data.time, "learned_policy_enabled" if enabled else "learned_policy_disabled",
                movement_mode["value"],
            )
            recorder.metadata["policy_checkpoint"] = str(policy_path) if enabled else None
            recorder.metadata["correction_capture"] = "teacher_targets_logged_for_policy_visited_states"
            recorder._write_metadata()
        teacher_override_box.configure(state="normal" if enabled else "disabled")
        policy_status.configure(
            text=("Policy active. CSV target columns keep scripted-teacher labels. "
                  "Use teacher override to correct the current policy state.")
            if enabled else (policy_load_error or "Policy disabled; scripted crawl controller is active.")
        )

    def toggle_teacher_override():
        if recorder.active:
            enabled = bool(teacher_override_var.get())
            recorder.add_command_event(
                data.time,
                "teacher_override_enabled" if enabled else "teacher_override_disabled",
                movement_mode["value"],
            )
        policy_status.configure(
            text=("Teacher correction is applied; recorded targets are teacher labels." if teacher_override_var.get()
                  else "Policy active. CSV target columns keep scripted-teacher labels.")
        )

    policy_available = policy_model is not None
    policy_checkbox = ttk.Checkbutton(
        policy_frame, text="Run the trained 10M policy during movement",
        variable=policy_enabled_var, command=toggle_policy_mode,
        state="normal" if policy_available else "disabled",
    )
    policy_checkbox.pack(padx=5, pady=(4, 0), anchor="w")
    teacher_override_box = ttk.Checkbutton(
        policy_frame, text="Apply scripted teacher targets now (correction)",
        variable=teacher_override_var, command=toggle_teacher_override, state="disabled",
    )
    teacher_override_box.pack(padx=5, pady=2, anchor="w")
    policy_status = ttk.Label(
        policy_frame,
        text=("10M policy loaded. Enable it, start an episode, then hold a direction." if policy_available
              else f"Policy unavailable: {policy_load_error or 'checkpoint failed to load'}"),
        wraplength=460,
    )
    policy_status.pack(padx=5, pady=(0, 4), anchor="w")
    ttk.Label(
        policy_frame,
        text="Hotkey E toggles teacher correction while a direction key stays held.",
        wraplength=460,
    ).pack(padx=5, pady=(0, 4), anchor="w")

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
        recorder.metadata["action_label_source"] = "scripted_teacher_targets"
        recorder.metadata["target_label_source"] = "scripted_teacher"
        recorder.metadata["target_label_semantics"] = (
            "target_*_deg contains teacher targets, independent of applied_controller"
        )
        recorder.metadata["learned_policy_active_at_start"] = bool(policy_enabled_var.get())
        recorder.metadata["policy_checkpoint"] = str(policy_path) if policy_enabled_var.get() else None
        recorder.metadata["correction_capture"] = (
            "teacher_targets_logged_for_policy_visited_states"
            if policy_enabled_var.get() else "scripted_demonstration"
        )
        recorder._write_metadata()
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
    teacher_key_latched = {"value": False}

    def key_down(event):
        key = event.keysym.lower()
        if key == "e" and policy_enabled_var.get() and not teacher_key_latched["value"]:
            teacher_key_latched["value"] = True
            teacher_override_var.set(not teacher_override_var.get())
            toggle_teacher_override()
            return "break"
        mode = key_modes.get(key)
        if mode is None:
            return
        if key not in pressed_keys:
            pressed_keys.append(key)
        start_gait(mode)
        return "break"

    def key_up(event):
        key = event.keysym.lower()
        if key == "e":
            teacher_key_latched["value"] = False
            return "break"
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


        teacher_targets = dict(targets)
        learned_policy_active = bool(policy_enabled_var.get() and policy_model and mode != "stop")
        applied_controller = "scripted_teacher"
        applied_targets = teacher_targets
        if learned_policy_active:
            if data.time + 1e-12 >= policy_state["next_inference_time"]:
                observation, _, _, _ = policy_runtime.observe(
                    policy_context, data, joint_ids, foot_geom_ids, mode
                )
                policy_history.append(observation)
                policy_state["targets"] = policy_runtime.angles_from_policy(
                    policy_context, observation
                )
                while policy_state["next_inference_time"] <= data.time + 1e-12:
                    policy_state["next_inference_time"] += DATASET_SAMPLE_PERIOD_SECONDS
            if not teacher_override_var.get():
                applied_targets = policy_state["targets"]
                applied_controller = "learned_policy"
        movement_mode["last_targets"] = dict(applied_targets)

        for joint_name, target in applied_targets.items():
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
                "applied_controller": applied_controller,
                "target_label_source": "scripted_teacher",
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
                label_targets = teacher_targets if learned_policy_active else applied_targets
                row[f"target_{joint_name}_deg"] = math.degrees(label_targets[joint_name])
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
