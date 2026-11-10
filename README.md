# Midori

Midori is a 12-servo quadruped spider robot project. The repository is organized so the MuJoCo simulator lives in its own folder and other robot, mobile, and model-training work can be added alongside it.

## Repository layout

```text
midori/
├── mujoco/
│   ├── control_test.py         Simulator UI and live controller
│   ├── gait.py                 Crawl gait math and joint layout
│   ├── episode_recorder.py     CSV and episode metadata writer
│   ├── goal_navigation_env.py  Goal-relative PPO environment
│   ├── phase2_wave_demo.py     Scripted greeting motion preview
│   ├── camera.py              MuJoCo front-camera capture
│   ├── hello.xml              Current 12-actuator spider model
│   ├── mjmodel.xml            Earlier prototype model
│   └── README.md              MuJoCo setup and controls
├── training/
│   ├── physics_policy_10m.py   Behavior-cloning and physics-informed trainer
│   ├── train_goal_ppo.py       Goal-navigation PPO trainer
│   ├── evaluate_policy.py      Locomotion checkpoint evaluator
│   └── README.md               Training instructions and status
├── pyproject.toml         Python dependencies
└── uv.lock                Locked Python dependencies
```

## Run the simulator

From the project root in PowerShell:

```powershell
uv sync
uv run python mujoco/control_test.py
```

The detailed simulator notes are in [mujoco/README.md](mujoco/README.md). The robot model is an approximation; measure the assembled printed frame before using it to derive physical servo commands.

## Planned multimodal robot model

The longer-term plan is a compact multimodal language model, estimated at about 150 million parameters, trained from scratch. The planned training sequence is pretraining, supervised fine-tuning, and then reinforcement learning. The intended data mix includes voice notes and commands, images, servo telemetry, and robot action sequences, so the robot can interpret its surroundings and choose actions. The current token budget estimate is approximately 5 billion tokens. Physics-informed neural networks are also under consideration for connecting learned predictions with robot dynamics.

The intended deployment is to run the model on a mobile phone and connect the phone to the robot's ESP32. Dataset construction, model architecture, training code, evaluation, safety constraints, and the mobile-to-ESP32 interface have not been implemented yet; these are project goals rather than current capabilities. See [mujoco/README.md](mujoco/README.md) for the current simulator status and next steps.
