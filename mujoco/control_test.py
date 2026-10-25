"""MuJoCo controller for the 12-servo spider prototype.

Run from the repository root with: uv run python mujoco/control_test.py
Motion is logged at a wall-clock rate, independent of viewer speed.
"""
import csv
import math
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "hello.xml"
LOG_DIR = HERE / "logs"
LOG_PERIOD_SECONDS = 0.1  # 10 rows per real-time second, even if simulation runs faster
MAX_ROWS_PER_LOG = 5000   # Start a new CSV after about 8 minutes 20 seconds
GAIT_FREQUENCY_HZ = 0.75
STRIDE_METRES = 0.035
SWING_FRACTION = 0.25
FOOT_LIFT_METRES = 0.020

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
    "front_left": (0.048, 0.034, math.pi / 4, 0.00),
    "front_right": (0.048, -0.034, -math.pi / 4, 0.50),
    "rear_left": (-0.048, 0.034, 3 * math.pi / 4, 0.25),
    "rear_right": (-0.048, -0.034, -3 * math.pi / 4, 0.75),
}

# Geometry from hello.xml. The rest pose is a modestly bent, four-foot stance.
COXA_LENGTH = 0.030
FEMUR_VECTOR = (0.035, 0.045)  # local X length, downward length at zero angle
TIBIA_VECTOR = (0.045, 0.055)
FEMUR_LENGTH = math.hypot(*FEMUR_VECTOR)
TIBIA_LENGTH = math.hypot(*TIBIA_VECTOR)
FEMUR_BASE_ANGLE = math.atan2(FEMUR_VECTOR[0], FEMUR_VECTOR[1])
TIBIA_BASE_ANGLE = math.atan2(TIBIA_VECTOR[0], TIBIA_VECTOR[1])
REST_FEMUR_ANGLE = -0.55
REST_TIBIA_ANGLE = 1.10
FOOT_RADIUS = 0.011


def rest_foot_geometry():
    """Return radial reach and downward reach for the defined rest joint pose."""
    total_femur_angle = REST_FEMUR_ANGLE
    total_tibia_angle = REST_FEMUR_ANGLE + REST_TIBIA_ANGLE
    femur_x = FEMUR_VECTOR[0] * math.cos(total_femur_angle) - FEMUR_VECTOR[1] * math.sin(total_femur_angle)
    femur_down = FEMUR_VECTOR[0] * math.sin(total_femur_angle) + FEMUR_VECTOR[1] * math.cos(total_femur_angle)
    tibia_x = TIBIA_VECTOR[0] * math.cos(total_tibia_angle) - TIBIA_VECTOR[1] * math.sin(total_tibia_angle)
    tibia_down = TIBIA_VECTOR[0] * math.sin(total_tibia_angle) + TIBIA_VECTOR[1] * math.cos(total_tibia_angle)
    return COXA_LENGTH + femur_x + tibia_x, femur_down + tibia_down


REST_RADIAL_REACH, REST_FOOT_DROP = rest_foot_geometry()
ROOT_STAND_HEIGHT = REST_FOOT_DROP + FOOT_RADIUS


class MotionRecorder:
    """Write bounded-size CSV chunks so a fast simulation cannot flood disk."""

    def __init__(self, joint_names, leg_names):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.joint_names = joint_names
        self.leg_names = leg_names
        self.part = 0
        self.rows_in_part = 0
        self._open_next_part()

    def _open_next_part(self):
        if hasattr(self, "file"):
            self.file.close()
        self.part += 1
        path = LOG_DIR / f"motion_{self.stamp}_part{self.part:03d}.csv"
        self.file = path.open("w", newline="", encoding="utf-8")
        columns = [
            "recorded_at_local", "sim_time_s", "movement_mode",
            "root_x_m", "root_y_m", "root_z_m",
            "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
            "root_vx_mps", "root_vy_mps", "root_vz_mps",
            "root_wx_radps", "root_wy_radps", "root_wz_radps",
        ]
        columns += [f"target_{name}_deg" for name in self.joint_names]
        columns += [f"actual_{name}_deg" for name in self.joint_names]
        columns += [f"actual_{name}_vel_degps" for name in self.joint_names]
        columns += [f"foot_{leg}_{axis}_m" for leg in self.leg_names for axis in ("x", "y", "z")]
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()
        self.file.flush()
        self.rows_in_part = 0

    def write(self, row):
        if self.rows_in_part >= MAX_ROWS_PER_LOG:
            self._open_next_part()
        self.writer.writerow(row)
        self.file.flush()
        self.rows_in_part += 1

    def close(self):
        self.file.close()


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


def crawl_targets(sim_time, mode):
    """Generate foot-space crawl targets, then solve each leg with 3-DoF IK."""
    targets = {}
    if mode == "stop":
        for _, name in JOINTS:
            targets[name] = REST_FEMUR_ANGLE if name.endswith("hip_pitch") else (
                REST_TIBIA_ANGLE if name.endswith("knee_pitch") else 0.0
            )
        return targets

    direction = -1.0 if mode == "backward" else 1.0
    turn_sign = -1.0 if mode == "left" else 1.0
    cycle = sim_time * GAIT_FREQUENCY_HZ

    for leg_name, (hip_x, hip_y, heading, phase_offset) in LEG_INFO.items():
        phase = (cycle + phase_offset) % 1.0
        if phase < SWING_FRACTION:
            swing_progress = phase / SWING_FRACTION
            stride_offset = (-0.5 + swing_progress) * STRIDE_METRES
            foot_lift = FOOT_LIFT_METRES * math.sin(math.pi * swing_progress)
        else:
            stance_progress = (phase - SWING_FRACTION) / (1.0 - SWING_FRACTION)
            stride_offset = (0.5 - stance_progress) * STRIDE_METRES
            foot_lift = 0.0

        if mode in ("forward", "backward"):
            travel_x = direction
            travel_y = 0.0
        else:
            # A positive left turn sweeps stance feet along the clockwise tangent
            # relative to the body; the ground reaction turns the chassis left.
            travel_x = turn_sign * math.sin(heading)
            travel_y = -turn_sign * math.cos(heading)

        foot_world_x = hip_x + REST_RADIAL_REACH * math.cos(heading) + stride_offset * travel_x
        foot_world_y = hip_y + REST_RADIAL_REACH * math.sin(heading) + stride_offset * travel_y
        relative_x = foot_world_x - hip_x
        relative_y = foot_world_y - hip_y
        local_x = math.cos(heading) * relative_x + math.sin(heading) * relative_y
        local_y = -math.sin(heading) * relative_x + math.cos(heading) * relative_y
        foot_down = REST_FOOT_DROP - foot_lift
        yaw, femur, tibia = inverse_kinematics(local_x, local_y, foot_down)

        targets[f"{leg_name}_hip_abduction"] = yaw
        targets[f"{leg_name}_hip_pitch"] = femur
        targets[f"{leg_name}_knee_pitch"] = tibia

    return targets


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

    recorder = MotionRecorder(joint_names, LEG_INFO)
    root = tk.Tk()
    root.title("Midori | Quadruped control")
    root.geometry("500x820")
    root.minsize(420, 560)

    ttk.Label(root, text="MIDORI | QUADRUPED CONTROL", font=("Segoe UI", 14, "bold")).pack(
        padx=12, pady=(12, 4), anchor="w"
    )
    ttk.Label(
        root,
        text="Hold a direction button or arrow key to move. Release it to stop.",
        wraplength=460,
    ).pack(padx=12, pady=(0, 6), anchor="w")

    movement_mode = {"value": "stop"}
    status = ttk.Label(
        root,
        text=f"Stopped. Motion log: mujoco/logs (10 samples/second; new file every {MAX_ROWS_PER_LOG} rows).",
        wraplength=460,
    )
    status.pack(padx=12, pady=(0, 8), anchor="w")

    def start_gait(direction):
        movement_mode["value"] = direction
        status.configure(text=f"Moving {direction}. Motion log: mujoco/logs (10 samples/second).")

    def stop_gait():
        movement_mode["value"] = "stop"
        status.configure(text="Stopped in the standing pose. Motion log: mujoco/logs (10 samples/second).")

    drive_frame = ttk.LabelFrame(root, text="Hold to move")
    drive_frame.pack(fill="x", padx=12, pady=(0, 8))
    drive_buttons = [
        ("Forward", "forward"),
        ("Backward", "backward"),
        ("Left", "left"),
        ("Right", "right"),
    ]
    for index, (label, mode) in enumerate(drive_buttons):
        button = ttk.Button(drive_frame, text=label)
        button.grid(row=index // 2, column=index % 2, sticky="ew", padx=4, pady=4)
        button.bind("<ButtonPress-1>", lambda _event, value=mode: start_gait(value))
        button.bind("<ButtonRelease-1>", lambda _event: stop_gait())
    ttk.Button(drive_frame, text="Stop", command=stop_gait).grid(
        row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4
    )
    drive_frame.columnconfigure(0, weight=1)
    drive_frame.columnconfigure(1, weight=1)

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

    next_log_wall_time = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 30
            viewer.cam.elevation = -45
            viewer.cam.distance = 0.42
            viewer.cam.lookat[:] = [0.0, 0.0, ROOT_STAND_HEIGHT * 0.45]
            root.protocol("WM_DELETE_WINDOW", root.destroy)

            while viewer.is_running():
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    break

                mode = movement_mode["value"]
                if mode == "stop":
                    targets = {
                        joint_name: math.radians(float(sliders[joint_name].get()))
                        for joint_name in joint_names
                    }
                else:
                    targets = crawl_targets(data.time, mode)

                for joint_name, target in targets.items():
                    data.ctrl[actuator_ids[joint_name]] = target

                mujoco.mj_step(model, data)
                viewer.sync()

                now = time.monotonic()
                if now >= next_log_wall_time:
                    row = {
                        "recorded_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "sim_time_s": data.time,
                        "movement_mode": movement_mode["value"],
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
                    recorder.write(row)
                    next_log_wall_time = now + LOG_PERIOD_SECONDS

            try:
                root.destroy()
            except tk.TclError:
                pass
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
