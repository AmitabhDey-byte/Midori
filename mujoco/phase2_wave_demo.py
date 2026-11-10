"""Preview a slow, three-foot-supported front-leg greeting motion in MuJoCo.

This is a reference motion for designing Phase 2 training. It is not yet a
learned policy and does not detect people or use camera images.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gait


MODEL_PATH = HERE / "hello.xml"
WAVE_LEG = "front_left"
RAISE_SECONDS = 1.6
WAVE_SECONDS = 4.0
LOWER_SECONDS = 1.6
REST_SECONDS = 2.0
COXA_SWEEP_RAD = math.radians(18.0)
WAVE_HZ = 0.9
WAVE_FEMUR_ANGLE = math.radians(-64.0)


def smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def wave_targets(elapsed: float) -> dict[str, float]:
    """Generate one eased greeting cycle while the other three legs stay planted."""
    period = REST_SECONDS + RAISE_SECONDS + WAVE_SECONDS + LOWER_SECONDS
    phase = elapsed % period
    targets: dict[str, float] = {}
    for _, joint_name in gait.JOINTS:
        if joint_name.endswith("hip_abduction"):
            targets[joint_name] = 0.0
        elif joint_name.endswith("hip_pitch"):
            targets[joint_name] = gait.REST_FEMUR_ANGLE
        elif joint_name.endswith("knee_pitch"):
            targets[joint_name] = gait.REST_TIBIA_ANGLE

    raise_start = REST_SECONDS
    wave_start = raise_start + RAISE_SECONDS
    lower_start = wave_start + WAVE_SECONDS
    raised = 0.0
    sweep = 0.0
    if raise_start <= phase < wave_start:
        raised = smoothstep01((phase - raise_start) / RAISE_SECONDS)
    elif wave_start <= phase < lower_start:
        raised = 1.0
        sweep = math.sin(2.0 * math.pi * WAVE_HZ * (phase - wave_start))
    elif lower_start <= phase:
        raised = 1.0 - smoothstep01((phase - lower_start) / LOWER_SECONDS)

    targets[f"{WAVE_LEG}_hip_pitch"] = (
        gait.REST_FEMUR_ANGLE
        + raised * (WAVE_FEMUR_ANGLE - gait.REST_FEMUR_ANGLE)
    )
    targets[f"{WAVE_LEG}_hip_abduction"] = sweep * COXA_SWEEP_RAD
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=24.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos = int(model.jnt_qposadr[root_joint_id])
    data.qpos[root_qpos:root_qpos + 3] = [0.0, 0.0, gait.ROOT_STAND_HEIGHT]
    data.qpos[root_qpos + 3:root_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

    actuator_ids: dict[str, int] = {}
    for _, joint_name in gait.JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_motor"
        )
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"Missing joint or actuator for {joint_name}")
        actuator_ids[joint_name] = int(actuator_id)
        if joint_name.endswith("hip_pitch"):
            angle = gait.REST_FEMUR_ANGLE
        elif joint_name.endswith("knee_pitch"):
            angle = gait.REST_TIBIA_ANGLE
        else:
            angle = 0.0
        data.qpos[int(model.jnt_qposadr[joint_id])] = angle
        data.ctrl[int(actuator_id)] = angle

    mujoco.mj_forward(model, data)
    frame_seconds = 0.02
    frame_skip = max(1, round(frame_seconds / model.opt.timestep))
    deadline = time.monotonic()
    print(
        "Phase 2 wave preview: front-left leg, three other legs held in the "
        "standing pose. This is a scripted reference, not a learned or "
        "person-detecting policy."
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -35
        viewer.cam.distance = 1.7
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        while viewer.is_running() and data.time < args.duration:
            targets = wave_targets(data.time)
            for joint_name, target in targets.items():
                data.ctrl[actuator_ids[joint_name]] = target
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            viewer.sync()

            deadline += frame_seconds
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()


if __name__ == "__main__":
    main()
