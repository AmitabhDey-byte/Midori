# Midori locomotion training

The `training/` folder contains the first learned locomotion policy and its Kaggle training path. It stays separate from `mujoco/`; the MuJoCo project remains the source of demonstrations and the physics model.

## Current dataset audit

The local episode set currently contains 110 complete episode folders, all recorded with `stance_transition_sequential_crawl_v3` at 50 Hz. There are 105 successful demonstrations and five non-success episodes (four falls and one interrupted turn). Successful counts are 26 forward, 27 backward, 27 left-turn, and 25 right-turn. The four tasks contribute 78,743 CSV rows in total. Metadata labels (`walk_forward`, `walk_backward`, `turn_left`, `turn_right`) are mapped to row command labels (`forward`, `backward`, `left`, `right`) before stratifying. Training uses only successful episodes for action imitation; failed and interrupted episodes are not treated as desired behavior. Splits are made by whole episode and stratified by task.

The data is still simulation-only, with no environment randomization. These episodes are demonstrations from the scripted crawl controller, not independent expert strategies. Closed-loop simulation is required: low offline target error alone does not establish stable walking. Add policy-generated recovery/correction episodes (DAgger-style) after the first rollout to teach behavior on states the current controller did not visit.

## Model in this run

`physics_policy_10m.py` trains a command-conditioned temporal Transformer with a measured **10,333,986 trainable parameters** (about 10.33M): four encoder blocks, width 512, eight attention heads, 1,344-wide feed-forward blocks, and 64 samples of history (1.28 s at 50 Hz). Its policy head predicts all 12 bounded joint targets. An auxiliary head predicts the next 18 generalized velocities and four nonnegative foot support forces.

This is **physics-informed / PINN-style training, not a full classical PINN**. The action objective imitates successful targets; a second objective predicts the next recorded velocities; and a gravity/contact residual applies vertical Newton's second law:

```text
m * (v_z[t+1] - v_z[t]) / dt = sum_i(Fz_i * contact_i) - m * g
```

Here `dt=0.02 s` is the CSV sample period, `g=9.81 m/s^2`, and the XML's total body mass is `0.828 kg`. Contact forces are constrained to be nonnegative and masked by logged foot contact. This does not enforce complete articulated-body dynamics, torque balance, or a friction cone; measured force/torque labels and a full robot dynamics model would be needed for those. The notebook should therefore not claim that the policy has learned all physics laws.

The feature input is the existing 54-value observation: five movement-command flags; 12 normalized joint angles and 12 joint rates; body height, orientation and linear/angular velocity; 12 body-frame foot coordinates; and four contact flags. The action output is 12 normalized servo position targets. The model sees robot feedback at each step, so actions respond to changing pose and velocity. Small continuous observation noise is added during training. This improves robustness to small measurement variation but does not replace closed-loop corrective demonstrations.

## Data and training calculations

- 50 Hz logs give `dt = 1 / 50 = 0.02 s` between CSV rows. MuJoCo itself advances at 0.002 s, so there are 10 physics substeps per recorded sample.
- 64 history rows span `64 / 50 = 1.28 s` of state/action context.
- A single 10.33M-parameter FP32 checkpoint is about `10,333,986 * 4 = 41,335,944 bytes`, approximately 39.4 MiB (optimizer state during training uses additional GPU memory).
- A standard dense-transformer training estimate for the policy alone is `6 * P * T` FLOPs per token, where `P=10.33M` and `T` is the number of training tokens/positions. This is only a rough compute estimate; attention, padding, optimizer work, and GPU utilization change actual run time.
- There are only 105 successful episodes, so the 10M model can overfit. Episode-level validation, early stopping, and the MuJoCo closed-loop check are required. More varied demonstrations and corrective rollouts are more useful than merely extending the same straight paths.

## Run it on Kaggle

1. Upload the prepared flat `D:\episodes\` folder as a private Kaggle Dataset. The files should be named `<episode_id>_episode.json` and `<episode_id>_state_action.csv` so all 110 pairs remain unique in one directory. The loader accepts both this flat naming and the original nested episode folders.
2. Upload `training/physics_policy_10m.py` as a second Kaggle Dataset, or put this file into a Kaggle Dataset named `midori-training-code`.
3. Import `kaggle_physics_policy_10m.ipynb` as a Kaggle Notebook and attach both datasets under **Add Input**. Update `EPISODES_ROOT` and `SCRIPT_PATH` in its settings cell to the paths Kaggle displays.
4. Select a GPU accelerator and run all cells. PyTorch, NumPy, and Python's standard library are used. The output is saved to `/kaggle/working/midori-output/`.
5. Download `best_policy_10m.pt` and `training_summary_10m.json`. The summary records accepted episode counts, task balance, parameter count, physics configuration, held-out error, and training time.

By default the script uses 80 epochs, 256-sample batches, early stopping after 10 stale validation epochs, and a 0.828 kg robot mass read from the current `mujoco/hello.xml`. If the robot XML/masses change, update `--mass-kg` in the notebook. The Kaggle split keeps each episode wholly in train, validation, or test. A weighted sampler balances the five per-frame command labels (stop, forward, backward, left, right), including command segments recorded inside an episode.

## Test in simulation

`evaluate_policy.py` accepts both the older GRU checkpoint and this Transformer checkpoint. After downloading the 10M checkpoint to `training/`, run a gravity-on forward rollout:

```powershell
uv run python training/evaluate_policy.py --checkpoint training/best_policy_10m.pt --duration 8 --settle 3
```

To see the rollout in the MuJoCo viewer, add `--viewer`. Evaluate other commands with `--command backward`, `--command left`, or `--command right`. The scripted comparison is forward-only; other commands run the learned policy alone. Assess displacement, falls, contact timing, and heading change as well as offline joint-angle error. Do not treat an offline MAE as proof of walking. Evaluator compatibility only means it can load and execute the model; it does not mean the checkpoint has passed a stability test.

## Later multimodal stages

Keep the locomotion policy as the fast control layer. A later voice/image model should turn audio and camera inputs into a bounded skill request (for example, `walk_forward` or `turn_left`) and the locomotion policy should execute it. Do not treat voice transcripts, image captions, or arbitrary robot datasets as 12-servo actions. They need task-appropriate labels and time alignment. Reinforcement learning and on-policy corrective data should follow only after a stable behavior-cloned rollout exists. The 10M locomotion network is not itself the planned multimodal LLM.
