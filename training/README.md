# Midori locomotion training

The `training/` folder contains the first learned locomotion policy and its Kaggle training path. It stays separate from `mujoco/`; the MuJoCo project remains the source of demonstrations and the physics model.

## Current dataset audit

The current local dataset contains **360 successful episodes** recorded with `stance_transition_sequential_crawl_v3` at 50 Hz: 90 each for forward, backward, left, and right. The latest training audit accepted all 360 episodes and all 247,034 teacher-labeled rows, with no missing CSVs, failed episodes, or gait-version mismatches. Metadata labels (`walk_forward`, `walk_backward`, `turn_left`, `turn_right`) are mapped to row command labels (`forward`, `backward`, `left`, `right`) before stratifying. The split is by whole episode and stratified by task: 248 train, 56 validation, and 56 test episodes.

The data is still simulation-only, with no environment randomization. These episodes are demonstrations from the scripted crawl controller, not independent expert strategies. Closed-loop simulation is required: low offline target error alone does not establish stable walking. Add policy-generated recovery/correction episodes (DAgger-style) after the first rollout to teach behavior on states the current controller did not visit.

## Teacher-label schema

New recordings use schema version 3. Each CSV row separates the controller that actually moved the robot (`applied_controller`) from the controller that supplied the supervised target (`target_label_source`). For DAgger-style correction, `applied_controller` can be `learned_policy` while `target_label_source` is `scripted_teacher`; the `target_*_deg` columns then contain the teacher's corrective joint targets. Do not treat the executed policy action as the teacher label.

The trainer accepts explicit teacher sources (`scripted_teacher`, `teacher`, or `expert`) and excludes rows explicitly tagged with another target source. Legacy CSVs without `target_label_source` remain supported: metadata teacher-source tags are honored, old CSVs with no controller column are treated as scripted demonstrations, and rows with an explicitly learned controller but no teacher-target declaration are excluded. The held-out analysis uses the same teacher-target mask as training.

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
- The 10M model can still overfit despite the larger episode set. Episode-level validation and closed-loop MuJoCo evaluation remain necessary; varied starting states and corrective rollouts are the next data improvements.

## Run it on Kaggle

1. Upload the prepared flat `D:\episodes\` folder as a private Kaggle Dataset. The files should be named `<episode_id>_episode.json` and `<episode_id>_state_action.csv` so each pair remains unique in one directory. The loader accepts both this flat naming and the original nested episode folders.
2. Upload `training/physics_policy_10m.py` as a second Kaggle Dataset, or put this file into a Kaggle Dataset named `midori-training-code`.
3. Import `kaggle_physics_policy_10m.ipynb` as a Kaggle Notebook and attach both datasets under **Add Input**. Update `EPISODES_ROOT` and `SCRIPT_PATH` in its settings cell to the paths Kaggle displays.
4. Select **GPU T4 x2** and run all cells. The notebook checks that two CUDA devices are visible; the training script uses `torch.nn.DataParallel` to split each global batch across both GPUs. The summary records GPU names/count and the parallel backend. PyTorch, NumPy, and Python's standard library are used. The output is saved to `/kaggle/working/midori-output/`.
5. Download `best_policy_10m.pt` and `training_summary_10m.json`. The summary records accepted episode counts, task balance, parameter count, physics configuration, held-out error, and training time.

By default the script uses 80 epochs, 256-sample batches, early stopping after 10 stale validation epochs, and a 0.828 kg robot mass read from the current `mujoco/hello.xml`. If the robot XML/masses change, update `--mass-kg` in the notebook. The Kaggle split keeps each episode wholly in train, validation, or test. A weighted sampler balances the five per-frame command labels (stop, forward, backward, left, right), including command segments recorded inside an episode.

## Test in simulation

`evaluate_policy.py` accepts both the older GRU checkpoint and this Transformer checkpoint. It runs with gravity on and virtual stabilization off. Run a gravity-on forward rollout:

```powershell
uv run python training/evaluate_policy.py --checkpoint training/best_policy_10m.pt --duration 8 --settle 3
```

To see the rollout in the MuJoCo viewer, add `--viewer`. Evaluate other commands with `--command backward`, `--command left`, or `--command right`. The scripted comparison is forward-only; other commands run the learned policy alone. Assess displacement, falls, contact timing, and heading change as well as offline joint-angle error. Do not treat an offline MAE as proof of walking.

### Latest checkpoint results

The current 10.33M checkpoint was trained for 80 epochs on the balanced 360-episode set. Kaggle reported two Tesla T4 GPUs using `torch.nn.DataParallel`; the run took about 80 minutes. The test mean joint-target MAE was **0.673 degrees**. The best validation loss was `0.0001691` at epoch 79. These are held-out imitation metrics on recorded states, not proof of closed-loop goal reaching or stable walking.

The separate goal-navigation PPO policy has been evaluated in MuJoCo. The long-range run completed 420,000 decisions; its latest evaluation reached the goal in 10 of 10 episodes. Its best checkpoint is `goal_nav_ppo_longrange/best/best_model.zip`. This is a high-level navigator over the scripted gait, separate from the 10.33M joint-target model.

Per-command held-out imitation metrics are in `per_command_test_metrics.json`; the local analysis command is:

```powershell
uv run python training/analyze_policy_commands.py --episodes D:\episodes --checkpoint training\best_policy_10m.pt
```

### First goal-navigation RL pilot

`mujoco/goal_navigation_env.py` randomizes goal direction, goal distance, and the robot's start heading in the bounded arena. The current long-range curriculum samples goals 0.45–1.15 m from the robot and allows 60 simulated seconds per episode. Its 59-value input includes goal distance and bearing relative to the robot, gravity, velocity, joint state, foot position/contact, and current command. The old fixed A/B/C/D cardinal buttons and markers are removed.

The PPO policy chooses among five commands (`stop`, `forward`, `backward`, `left`, `right`) every 0.2 simulated seconds. The existing crawl gait generator converts each command into the 12 servo targets. This is a hierarchical pilot: PPO learns which direction command moves Midori toward a randomized goal; it does **not** fine-tune the 10.33M joint-target model or learn the low-level crawl from scratch. The locomotion checkpoint is not loaded by this trainer.

Run a short pilot from the repository root:

```powershell
uv sync
uv run python training/train_goal_ppo.py --timesteps 20000 --envs 1 --eval-freq 5000 --eval-episodes 5
```

The MLP PPO policy runs on CPU; MuJoCo also runs locally. One environment avoids Windows multiprocessing startup overhead for this first run. The run may take a few minutes or longer depending on the machine because every transition advances MuJoCo physics. Logs and checkpoints go to `training/goal_nav_ppo/`; the best evaluation model is `training/goal_nav_ppo/best/best_model.zip`, and the final model is `training/goal_nav_ppo/goal_nav_ppo_final.zip`. Review evaluation success, mean reward, distance-to-goal, falls, and boundary exits before scaling up. This pilot tests whether the reward, observations, and scripted gait let PPO learn goal-relative command selection; it is not yet autonomous low-level locomotion learning.

Continue from the current best checkpoint for 300,000 additional decisions with evaluations every 30,000 decisions. The resumed trainer keeps the checkpoint and writes a separate output folder:

```powershell
uv run python training/train_goal_ppo.py --resume training/goal_nav_ppo_curriculum/best/best_model.zip --timesteps 300000 --envs 1 --eval-freq 30000 --eval-episodes 10 --checkpoint-freq 30000 --output training/goal_nav_ppo_longrange
```

With resume enabled, `--timesteps` is the number of additional environment decisions. The output folder receives periodic checkpoints, evaluation logs, the best checkpoint, and the final checkpoint. The long-range run completed 420,000 cumulative decisions. Keep earlier runs for comparison.

Watch the best checkpoint in the MuJoCo viewer:

```powershell
uv run python training/play_goal_ppo.py --episodes 3
```

Close the MuJoCo viewer to stop playback. Use `--seed 41` to reproduce the first goal sequence, `--model training/goal_nav_ppo/goal_nav_ppo_final.zip` to view the final checkpoint, or `--fast` to skip real-time pacing.

### Capture corrective episodes

The MuJoCo control panel now has **Run the trained 10M policy during movement** and **Apply scripted teacher targets now (correction)**. Start the policy, start a dataset episode, then hold backward or a turn. If the robot starts to drift or tip, enable teacher correction while keeping the same direction active. The simulation applies the scripted gait, while the CSV stores its joint targets as labels for the policy-visited states. Release and end the episode after the teacher has recovered the motion. Mark `success` only if the episode finishes upright and moving as intended; mark falls as failures. Save at least 8–10 corrected episodes for each weak direction, with varied initial poses and times at which correction is applied. Keep the policy checkbox on for these episodes so the observations include states it actually visits.

After capture, add the new episode folders to the dataset and retrain. The existing trainer will treat their teacher targets as imitation labels when the outcome is `success`.

## Later multimodal stages

Keep the locomotion policy as the fast control layer. A later voice/image model should turn audio and camera inputs into a bounded goal or skill request and the locomotion policy should execute it. Do not treat voice transcripts, image captions, or arbitrary robot datasets as 12-servo actions; they need task-appropriate labels and time alignment. The goal-conditioned RL policy is a separate navigation stage, while the 10M locomotion network is not itself the planned multimodal LLM.
