"""Front-camera capture for the MuJoCo spider model.

Example:
    uv run python mujoco/camera.py --output mujoco/camera_frames/front.png

The RobotCamera class can also be used from a simulation loop. capture(data)
returns an RGB uint8 NumPy image without writing a file.
"""
import argparse
from datetime import datetime
from pathlib import Path

import mujoco
from matplotlib import image as mpl_image

try:
    from .control_test import (
        JOINTS,
        MODEL_PATH,
        REST_FEMUR_ANGLE,
        REST_TIBIA_ANGLE,
        ROOT_STAND_HEIGHT,
    )
except ImportError:
    from control_test import (
        JOINTS,
        MODEL_PATH,
        REST_FEMUR_ANGLE,
        REST_TIBIA_ANGLE,
        ROOT_STAND_HEIGHT,
    )

CAMERA_NAME = "front_camera"
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240


class RobotCamera:
    """Render images from the chassis-mounted MuJoCo camera."""

    def __init__(self, model, camera_name=CAMERA_NAME, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.camera_name = camera_name

    def capture(self, data):
        """Return an RGB image array for the current simulation state."""
        self.renderer.update_scene(data, camera=self.camera_name)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def make_standing_state(model):
    """Create the same splayed standing pose used by the control window."""
    data = mujoco.MjData(model)
    data.qpos[2] = ROOT_STAND_HEIGHT
    for _, joint_name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in model: {joint_name}")
        angle = REST_FEMUR_ANGLE if joint_name.endswith("hip_pitch") else (
            REST_TIBIA_ANGLE if joint_name.endswith("knee_pitch") else 0.0
        )
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)
    return data


def main():
    parser = argparse.ArgumentParser(description="Capture one frame from the MuJoCo front camera.")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: mujoco/camera_frames/front_TIMESTAMP.png)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = make_standing_state(model)
    camera = RobotCamera(model, width=args.width, height=args.height)
    try:
        frame = camera.capture(data)
    finally:
        camera.close()

    output = args.output or (
        Path(__file__).resolve().parent
        / "camera_frames"
        / f"front_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_image.imsave(output, frame)
    print(f"Saved {frame.shape[1]}x{frame.shape[0]} front-camera frame to {output}")


if __name__ == "__main__":
    main()
