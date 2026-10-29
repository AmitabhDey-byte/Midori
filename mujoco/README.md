# Midori MuJoCo Work

## Purpose

This folder contains the current MuJoCo prototype for Midori, a four-legged spider-style robot intended to use twelve servo joints. The simulator is a place to test joint naming, control ideas, and logging before connecting the mobile model and ESP32. It is not yet a dimensionally accurate digital twin of the printed robot.

## Run

Run commands from the repository root (`D:\midori`):

```powershell
uv sync
uv run python mujoco/control_test.py
```

The program opens a MuJoCo viewer and a separate control window. Physics advances at the model's 2 ms timestep against a wall-clock accumulator, with multiple physics steps between viewer refreshes, so one simulation second is approximately one real second when the computer keeps up. Close either window to end the session. Dataset files are created only when you start an episode in the control window; see [Recording dataset episodes](#recording-dataset-episodes).

## Front camera

`hello.xml` includes a small camera module mounted at the front of the chassis. The MuJoCo camera is named `front_camera`, points forward along the robot's X axis, and tilts down slightly toward the floor. The camera shell is represented on the chassis as a small dark box.

Capture a single PNG from the standing pose:

```powershell
uv run python mujoco/camera.py --output mujoco/camera_frames/front.png
```

The default output is a 320 by 240 image in `mujoco/camera_frames/`; this folder is ignored by Git. Increase the output size with `--width` and `--height` up to the MuJoCo offscreen framebuffer limit. The capture script saves one image per run; it does not stream images or create a sequence automatically.

For use inside a future perception loop, import `RobotCamera` and call `capture(data)` after advancing the MuJoCo simulation:

```python
import sys
sys.path.insert(0, "mujoco")
from camera import RobotCamera

camera = RobotCamera(model, width=320, height=240)
rgb_frame = camera.capture(data)  # NumPy uint8 array, shape (240, 320, 3)
camera.close()
```

This is a virtual simulation camera. A physical camera module, its connection to the phone or ESP32, and image transport to the planned multimodal model remain future hardware/software work.

## Controls

The Drive panel uses hold-to-move controls:

- Hold Forward or Backward to move; release the button to stop the gait and return to the joint pose you set with the sliders.
- Hold Turn Left or Turn Right to rotate; release the button to stop turning and hold that same manual pose.
- The arrow keys and WASD do the same while the Tkinter control window has keyboard focus. Releasing the key stops that motion.
- Stop ends the gait and returns to the manual slider pose captured when movement began.
- Stop and return to standing pose explicitly sets all joint targets to the model's default standing pose.

The standing pose splays the femurs away from the chassis and points the tibias down to the floor, giving the robot a low, crab-like stance. The default viewer camera is set to show the legs from above and from one side.

Gravity is `-9.81 m/s²` and is active in every mode. `Optional virtual pose stabilization` is off by default. If enabled, it applies an external spring-damper wrench to the chassis while stopped and therefore overrides natural falling; leave it off for real-time gravity/contact behavior and for dataset episodes that should represent unassisted physics. It is not a physical actuator or hardware balance controller.

The gait is a continuous, one-foot-at-a-time four-beat crawl based on the supplied Arduino reference. It assigns a quarter-cycle offset to each leg and interpolates the chassis target toward the nominal triangle formed by the supporting feet. Each swing moves its foot forward and up; each stance sweeps it backward against the floor. The trajectories use quintic easing to avoid abrupt reversals. Forward, backward, and turn commands use a 40 mm stance/swing path. The nominal swing lift is 30 mm over a 2 mm stance preload (28 mm net lift), and the cycle is 0.85 Hz. When movement starts from the deeply crouched slider pose shown in the shared screenshot, the joints first ease over 1.2 seconds into a walk-ready stance (femur -45°, tibia +65°). That gives the IK room to lift and plant each claw without commanding beyond the joint stops. Releasing the direction eases back to the manual slider pose. The gait clock resets on a new direction so movement starts from a predictable foot phase. Left and right use opposite stance sweeps. Episode metadata records gait and actuator settings, including the transition stance, so new data can be distinguished from older recordings. Printed STL meshes are visual-only; hidden toe spheres are positioned at the actual pointed claw tips, with narrower tibia collision capsules so the tips contact the floor first.

The virtual position servos use a 0.45 Nm torque limit for clearer joint motion in simulation. This is a visual-control setting, not a measured SG90 torque model; replace it with measured servo torque, speed, and load before treating the simulation as hardware-calibrated.

### Video movement reference

The gait and stance were compared with the [Spider Robot v3.0 demo](https://youtu.be/5hAmD499sJs). The visible robot has a low central body, legs spread outward, and narrow pointed feet. Its leg movement is a crawl with individual feet lifting and returning to the floor, rather than a wheeled or holonomic glide. The original demonstration uses finite step sequences, while Midori keeps the requested hold-to-move control: it continues while a direction is held and stops on release.

This remains an initial simulator gait, not a calibrated physical-robot controller. The model uses estimated dimensions and contact properties. It can still slide, tip, or walk imperfectly; tune it against the exact printed frame before transferring commands to hardware. Use the sliders for individual joint inspection.

## Recording dataset episodes

The controller records only when an episode is started from the `Dataset episode capture` panel. Select a task, select the result at the end, press `Start episode`, demonstrate the task with the hold-to-move controls, then press `End episode and save`. Use one episode for one short task attempt; for example, select `walk_forward`, record a short forward command and release, then mark its outcome.

Each episode is saved to its own folder:

```text
mujoco/logs/episodes/sim_<timestamp>/
  episode.json
  state_action.csv
```

Rows are sampled every 0.02 simulation seconds (50 Hz nominal), rather than every fixed wall-clock interval. A 10-second simulation episode is about 500 rows. `episode.json` stores the task, simulation source, schema version, simulation timestep, outcome, duration, and timestamped command press/release events. If the app closes during recording, the episode is finalized with outcome `interrupted`.

`state_action.csv` stores episode and step IDs; episode/simulation timestamps; movement mode; whether virtual balance assist is active; body position, quaternion, linear and angular velocity; target and simulated actual angle/velocity for all 12 joints; world-space foot XYZ; and a per-foot floor-contact flag. Episode metadata records whether manual-pose balance assist was enabled and command events record changes to it. The `target_*` columns are commands sent to MuJoCo. The `actual_*` columns are simulated joint positions, not physical servo encoder readings. Every episode is marked `mujoco_simulation`; do not mix these values with hardware data without a separate `source` label.

The `logs/` folder is ignored by Git so run data is not accidentally committed. Back up episodes you want to keep in the dataset directory before cleaning or regenerating logs.

### First pilot collection

Start with five short episodes for each of `stand_balance`, `walk_forward`, `walk_backward`, `turn_left`, and `turn_right` (25 episodes total). Keep each attempt around 5–10 simulated seconds. Select one task before recording, start the episode, issue only that task with the hold controls, release to stop, select the correct outcome, and save. Use `mixed_command_sequence` only after the single-command episodes look correct. Do not discard a fall or bad turn; mark its outcome so it can be filtered or studied later.

Inspect a few CSVs before collecting more: episode time should increase in 0.02-second steps, command events should align with the intended controls, target and actual angles should differ only when the simulated joint lags, and foot-contact flags should change as feet lift and land. At 50 Hz, five seconds gives about 250 rows. The first pilot validates recording and labels; it is not yet enough to claim a robust trained policy.

## Current model

`hello.xml` contains a free chassis and four three-joint legs. The control names are retained for compatibility with the first slider controller; their anatomical meaning in this model is:

| Existing control name | Model role |
| --- | --- |
| `*_hip_abduction` | Coxa yaw about the vertical axis |
| `*_hip_pitch` | Femur pitch |
| `*_knee_pitch` | Tibia/knee pitch |

The model uses gravity, a ground plane, and position actuators for the 12 joints. The supplied STLs are copied into `mujoco/assets/` and used for the layered body, coxa brackets, femur links, hinge brackets, and long pointed tibia/claws. The four coxa meshes sit on the yaw links, the femur meshes follow hip pitch, and the red tibia meshes follow the knee joints. The displayed body and leg contours now come from the provided parts instead of generic wheel-like or capsule-only visuals. A compact collision core and limb proxies retain stable contact behavior; the decorative STL meshes do not collide.

The mesh dimensions are now available from the files, but their final pivots, servo neutral positions, masses, center of mass, joint stops, and collision hulls are still estimates. Mesh orientations and mount offsets are approximated from the supplied assembly photo and the shared CAD coordinate frame. Do not treat this visual update as measured servo calibration.

All four femur-pitch joints now allow a simulator minimum of -86° (with the prior +60° upper limit). Their position actuators use the same range so the slider can reach that endpoint. This is a simulation setting; verify the printed bracket and servo hard stop before copying it to the physical robot.

### Assumptions that still need measurements

The product is a printed-parts kit sold without servos. The seller identifies two versions for different servo-horn sizes. The provided STL bounds are about 84 by 76 mm for the body plates, 38 by 39 mm for the coxa parts, about 50 mm for the femur, and 100 mm long for each pointed tibia. The kinematic pivot-to-pivot distances are not identical to part bounding-box lengths and still need measurement. Servo model, mass, and calibrated joint ranges are not specified.

Before treating the simulation as a twin, measure the actual frame and record:

1. Chassis length, width, thickness, and assembled height.
2. Pivot-to-pivot lengths for the coxa, femur, and tibia.
3. Servo make/model and whether the printed parts fit original-size or larger clone horns.
4. Servo neutral positions, rotation directions, hard stops, and safe operating limits.
5. Printed frame mass, total assembled mass, and approximate centre of mass.

See [robot_spec.md](robot_spec.md) for the measurement checklist.

## Project direction: multimodal model and robot control

The intended later project is a multimodal language model, approximately 150 million parameters, pretrained from scratch and ultimately run on a mobile phone connected to the robot's ESP32. The proposed sequence is:

1. Build a custom training dataset and tokenizer/representation suitable for the chosen text, audio, and image inputs.
2. Pretrain from scratch using an estimated mixture of about 5 billion tokens. Planned data includes voice notes and commands, images, servo readings, and action sequences. Examples will need time alignment so a command, observed scene, robot state, and resulting action can be learned together.
3. Supervised fine-tune the pretrained model for instruction following and grounded robot tasks.
4. Explore reinforcement learning after a reliable simulator and evaluation tasks exist.
5. Investigate physics-informed neural networks or other physics-aware methods for predicting dynamics and constraining action selection.
6. Build a phone-to-ESP32 interface with an explicit command protocol, telemetry, and limits on allowed servo commands.

These are plans, not implemented features. There is currently no multimodal model, custom dataset, pretraining run, SFT or RL pipeline, physics-informed network, mobile inference app, or ESP32 communication code in this folder. The 150M parameter size and 5B token estimate are preliminary project targets; architecture, data rights and quality, compute, evaluation, and deployment performance still need to be worked out.

## Current simulator preview

The screenshot below shows the updated STL-based model in its standing pose. It uses the supplied body frames and articulated printed leg shapes with pointed red claw meshes. Simplified hidden collision proxies remain for simulation stability.

![STL-based MuJoCo spider model](screenshot.png)

## Suggested next simulator work

1. Measure servo pivot-to-pivot distances and refine the STL mount transforms.
2. Calibrate each simulated joint's zero direction and range against the assembled robot.
3. Refine the collision hulls and test floor contact with the measured assembled mass.
4. Compare each motion log with real ESP32 telemetry after that interface exists.
5. Define repeatable simulation and hardware evaluation tasks before using reinforcement learning.
