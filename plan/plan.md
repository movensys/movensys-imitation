# movensys-imitation: project plan

Compare ACT, Diffusion Policy and Flow Matching on PushT with the Dobot CR3A. Demonstrations come from Meta Quest teleoperation through MoveIt Servo. Commands reach the arm through WMX-R2, first in Isaac Sim, then on the real robot.

| | |
|---|---|
| Version | 2.0, 2026-10-13. Merges `movensys_imitation_plan.md` (v1.1) and `Simplified_plan.md`; both are archived under `plan/archive/`. |
| Host | Ubuntu 24.04, ROS 2 Jazzy, Python 3.12, RTX 5090, Isaac Sim 5.1 (`~/isaac-sim`), WMX3 SDK (`/opt/wmx3`) |
| Start | 2026-10-13, 10 weeks, one engineer (7 to 8 weeks with a controls engineer **C** and an ML engineer **M** in parallel) |

**Changes from v1 (review and accept or revert each):** T block scaled to 40 mm per unit so a 40 mm tag fits on it (v1 geometry had no room for the tag); overhead camera at 1280 x 720 so tags are large enough to detect from 0.7 m; real-robot checkpoint chosen by validation loss, not by sim score; 30 real trials per policy instead of 20; the guard arbitrates between teleop and policy input by mode; Phase 0 given two weeks and Phase 2 shortened by one; optional extras cut to a single list.

---

## 1. Goal and done criteria

**Goal.** Measure, on the same task and robot, how the three policies differ in success rate, motion quality and inference cost, and leave behind a pipeline that can be pointed at the next task.

**Deliverables.**

1. The `movensys-imitation` repository: Python library (policies, PushT geometry, scoring, dataset tools), ROS 2 package (guard, recorder, episode manager, scorer, policy runner), Isaac scene assets, printable hardware files, setup docs.
2. A Flow Matching policy registered in LeRobot that shares the Diffusion Policy encoder and UNet, so the only variable is the generative objective.
3. Three datasets (gym-pusht converted, PushT-CR3A-sim, PushT-CR3A-real) and checkpoints for three policies x three seeds on each.
4. Results tables (Section 4) with videos and a failure taxonomy.

**Done when** all three policies train and evaluate on gym-pusht in the published range, push the T in Isaac Sim on a fixed set of 50 start states, run on the real CR3A through WMX-R2 on a fixed set of 30 start marks, and a second engineer can reproduce a run from the docs.

**Out of scope.** Other tasks, reinforcement learning, Jetson deployment, new WMX3 features, VLA fine-tuning as a main line.

---

## 2. Stages and milestones

```
Phase 0  Environment        venv, containers, Quest, WMX streaming decision           M0
Phase 1  Gym-PushT          Flow Matching implemented, 3 policies benchmarked         M1
Phase 2  Isaac Sim          PushT scene, teleop, recorder, runner, sim evaluation     M2
Phase 3  Hardware-in-loop   WMX simu platform drives Isaac Sim through WMX-R2        M3
Phase 4  Real CR3A          real demos, training, 30 trials per policy               M4
Phase 5  Results            tables, failure taxonomy, docs                           M5
```

Each stage reuses the previous stage's code. Autonomous hardware runs start only after the streaming controller, soft limits, E-stop, watchdog and low-speed motion are verified (Phase 4 checklist).

### Master tracker

Update at every review. Status: `Not started`, `In progress`, `Blocked`, `Done`. Evidence: a link to the CSV, log, video or PR that proves it.

| ID | Milestone | Target week (dates) | Status | Evidence |
|---|---|---|---|---|
| M0 | Environment, Quest teleop in sim, WMX streaming decision | 2 (Oct 13 to 24) | Done 2026-10-15 | P0.1 to P0.7 status cells; `doc/1_setup.md` sections 1.3 to 1.6 pending (P0.8) |
| M1 | Three policies benchmarked on gym-pusht, configs frozen | 3 (Oct 27 to 31) | Not started | |
| M2 | Sim dataset collected, three policies evaluated on 50 start states | 6 (Nov 17 to 21) | Not started | |
| M3 | Policy episode through WMX-R2 on the simu platform, controller merged | 6 (Nov 17 to 21) | Not started | |
| M4 | Real-robot evaluation, 30 trials x 3 policies | 9 (Dec 8 to 12) | Not started | |
| M5 | Results document, videos, setup docs | 10 (Dec 15 to 19) | Not started | |

| Week | Dates | Focus |
|---|---|---|
| 1 to 2 | Oct 13 to 24 | Phase 0 (M1 work starts in week 2 if an ML engineer is available) |
| 3 | Oct 27 to 31 | Phase 1 benchmark, freeze configs (M1) |
| 4 | Nov 3 to 7 | Phase 2: scene, guard, recorder, pilot episodes |
| 5 | Nov 10 to 14 | Phase 2: collection, training, runner |
| 6 | Nov 17 to 21 | Phase 2 evaluation (M2), Phase 3 HIL (M3) |
| 7 | Nov 24 to 28 | Phase 4: hardware, soft limits, cameras, low-speed check |
| 8 | Dec 1 to 5 | Phase 4: real collection, training |
| 9 | Dec 8 to 12 | Phase 4 evaluation (M4) |
| 10 | Dec 15 to 19 | Phase 5 (M5) |

A blocked prerequisite takes priority over starting the next phase. There is no slack in this schedule; if Phase 0 slips past week 2, cut seeds from 3 to 2 on real data before cutting anything else.

---

## 3. Phases

Owner **C** = controls / robotics, **M** = ML. One person does the tasks in order; C and M tasks inside a phase are independent.

### Phase 0: environment and prerequisites (weeks 1 to 2)

Goal: prove the existing stack, the ML environment and the WMX streaming path work before writing new code.

| ID | Task | Owner | Done when | Status |
|---|---|---|---|---|
| P0.1 | Create the repo layout (Section 5), MIT licence, lint workflow, `.gitignore` for `datasets/ outputs/ .venv/ wandb/`. | M | CI green on an empty package | Done 2026-10-13 (ruff, ament_flake8, ament_xmllint, pytest pass locally; CI runs on first push) |
| P0.2 | Build the venv (Appendix A) and run the smoke test: 500 steps of Diffusion on `lerobot/pusht`, then eval 5 episodes. | M | `torch.cuda.is_available()` true, eval prints a success rate, `import rclpy` works in the venv | Done 2026-10-13 (torch 2.11+cu128, LeRobot 0.6.1; six environment issues fixed, see `doc/1_setup.md`) |
| P0.3 | Switch `movensys-manipulator` to `origin/devel`, add `ros-jazzy-moveit-servo` to the `isaac-ros_4.1` Dockerfile, build and start the manipulator container with `ROS_DOMAIN_ID=70` from the host shell. | C | `ros2 service type /servo_node/switch_command_type` returns `moveit_msgs/srv/ServoCommandType` | Done 2026-10-15 (reported at review) |
| P0.4 | Open `3a_trajectory_simulation.usd` in Isaac Sim, start `sim_bridge` and `moveit` with `use_sim_time:=true`, move the arm with `keyboard_teleop` in `p` mode. | C | Arm moves in Isaac from the keyboard | Done 2026-10-15 (`sim_bridge` with `simulator:=isaacsim` and `moveit` with `use_sim_time:=true` observed running) |
| P0.5 | Build the teleop container, install the CloudXR runtime from NGC, run `quest_pose.launch.py`, connect the Quest, then run `quest_servo_teleop.launch.py motion_mode:=planar` in sim. | C | `/quest_pose_publisher/controller_pose_right` at about 55 Hz; arm follows the hand in Isaac | Done 2026-10-15 (`quest_pose_node` with `hand:=both pose_source:=aim` and the CloudXR runtime observed running) |
| P0.6 | **Streaming controller decision.** Compare `joint_position_controller`, `servo_stream_controller` (wmx-r2 `feature/api-buffer-command-streaming*`), the wmx-r2 cyclic-buffer branch and the two manipulator-side cyclic-buffer branches (`feature/cyclic-buffer-for-moveit-servo`, `feature/moveit-servo-cyclic-buffer`). Pick one, rebase on `main`, open a PR. Fix the gripper node's hard-coded `/opt/lmx/` path. | C | PR open; a one-page comparison note in `doc/1_setup.md` | Done 2026-10-15 (reported at review; add the chosen controller and PR link here) |
| P0.7 | WMX simu platform: extend `simu_network.def` to six axes, launch `wmx_r2_cr3a_manipulator.launch.py` with the chosen controller, publish a hand-made `JointTrajectory`. | C | `/joint_states` follows the trajectory | Done 2026-10-15 (reported at review) |
| P0.8 | Write `doc/1_setup.md`: venv, containers, domain ID, sim time, CloudXR, the `sudo --preserve-env` line for wmx-r2. | C | A second person reproduces P0.2 to P0.7 from the doc | In progress: sections 1.3 to 1.6 of `doc/1_setup.md` are still placeholders |

Main blocker: WMX-R2 streaming. The chosen controller also shapes the smoothness metric (one interpolation per 40 Hz Servo message gives stop-and-go motion), so the choice must be fixed before any comparison and reported with the results.

### Phase 1: Flow Matching policy and gym-pusht benchmark (weeks 2 to 3)

Goal: a reliable benchmark and a tested Flow Matching policy before any robot-specific work.

| ID | Task | Owner | Done when | Status |
|---|---|---|---|---|
| P1.1 | Implement `flow_matching_unet` (Section 6) with unit tests: shapes, loss decreases on synthetic data, 1-step and 16-step samplers finite and seed-deterministic, 2-D toy velocity field matches Meta's `flow_matching` library. | M | Tests pass | Done 2026-10-14: `movensys_imitation/policies/flow_matching_unet/`, 15 tests pass (`test_flow_matching_unet.py`) |
| P1.2 | `movensys-train` and `movensys-eval` entry points that register the policy and call LeRobot's mains; YAMLs for the three policies. | M | `--policy.type=flow_matching_unet` resolves | Done 2026-10-14: `configs/{pusht_gym,diffusion,flow_matching,act}.yaml`, comma-separated `--config_path`; 300-step smoke train, reload and eval of `flow_matching_unet` passed |
| P1.3 | Reference run: Diffusion on `lerobot/pusht`, 200 k steps, 50 eval episodes. | M | Success within the 65 % band of the model card | Done 2026-10-14: 58 % final eval (64 % at the step-200k in-training eval), 50 episodes, 4.5 h on one GPU; `scripts/ref_diffusion_pusht.sh`, outputs in `outputs/ref_diffusion_pusht/` |
| P1.4 | Benchmark: ACT (default horizon and matched 16/8, with and without VAE), Diffusion (DDIM 10, DDPM 100 at inference), Flow Matching (Euler 1, 2, 4, 8, 16 at inference). 100 k steps, batch 64, 3 seeds, 50 fixed episodes each. Record train time, peak VRAM, latency mean and p95. | M | Table R1 filled | Running since 2026-10-14: `scripts/benchmark_gym.sh` (`movensys-benchmark`), 18 trainings, about 35 GPU hours; CSVs land in `results/gym/` |
| P1.5 | `compare.py` builds tables and plots from the eval CSVs; freeze hyperparameters for the robot phases. | M | Frozen configs committed | `movensys-compare` written and tested on a smoke matrix 2026-10-14; freeze after P1.4 finishes |
| P1.6 | Convert `lerobot/pusht` into the recorder's schema (Section 6, pixel units) to prove the feature definitions train. | M | One training run on the converted set | Done 2026-10-14: `movensys-convert-gym` (`data/convert_gym_pusht.py`), Diffusion trains on the converted set (`doc/3_gym_pusht.md`) |

Exit: three policies train and evaluate in one harness; Diffusion reproduces the published range; Flow Matching with 4 to 8 Euler steps is within noise of Diffusion; configs frozen and not re-tuned afterwards.

### Phase 2: PushT in Isaac Sim (weeks 3 to 6)

Goal: the whole collection and inference pipeline on the simulated CR3A with ground-truth block poses.

| ID | Task | Owner | Done when | Status |
|---|---|---|---|---|
| P2.1 | Scene `9a_pusht_simulation.usd` from `3a`: T block, target sheet, pusher tip under `Link6`, overhead camera from `7b` at 1280 x 720 and 30 Hz on `/image_top/rgb`, T pose publisher `/pusht/t_block/pose`, T teleport subscriber `/pusht/t_block/pose_cmd`. Reach check over the whole push area at push height with the tool down; fix the area centre. | C | Teleop reaches all four corners without Servo singularity slow-down | |
| P2.2 | `planar_guard` with tests (z and orientation lock, workspace clamp, step limit, EEF-distance limit, watchdog, Servo POSE switch, source arbitration by mode) and `task.yaml`, `guard.yaml`. | C | Tests pass; teleop runs through the guard | |
| P2.3 | `pusht_scorer` (sim mode) and `tasks/pusht/scoring.py` (shapely coverage), tests against hand-computed cases. | M | Tests pass | |
| P2.4 | `episode_recorder`, `episode_manager`, `collect.launch.py mode:=sim`. Pilot 10 episodes; check with `lerobot-dataset-viz`; replay check (recorded actions through the guard reproduce the final T pose); reset repeatability check (10 teleports, T velocity zero after each). | M + C | Pilot dataset passes all three checks | |
| P2.5 | Operator tuning: `position_scale`, guard step, crop, tip height. Write the operator guide in `doc/4_sim.md`. | C | Operator reaches 0.95 coverage in under 25 s on most tries | |
| P2.6 | Collect 150 to 200 episodes on seeds 1000+; keep successes; log the discard rate. | C | Dataset `movensys/pusht_cr3a_sim_v1` with stats | |
| P2.7 | Train the three policies, 3 seeds, frozen configs, while the simulator is idle. | M | 9 checkpoints | |
| P2.8 | `policy_runner` with `shadow`, `slow`, `normal` modes; `run_policy.launch.py mode:=sim`. | M | First autonomous success in sim | |
| P2.9 | `eval_sim.py`: 50 fixed start states x 3 policies x 3 seeds, CSV and video per episode; Flow Matching step-count ablation. | M | Table R2 filled | |

Exit: measured sim success rates on the fixed set; collect and run launch files work with `mode:=sim`.

### Phase 3: hardware-in-the-loop (week 6, overlaps P2.9)

Goal: validate the real streaming path with no hardware at risk.

| ID | Task | Owner | Done when | Status |
|---|---|---|---|---|
| P3.1 | `9b_pusht_hil.usd` (`/isaacsim` namespace). Launch: Isaac `9b`, wmx-r2 on the simu platform with the streaming controller, `/joint_states` to `/isaacsim/joint_command` relay, manipulator container with `sim_bridge` disabled, guard, runner. | C | Arm in Isaac follows WMX-reported joints | |
| P3.2 | 10 teleop and 10 policy episodes through this path. Measure joint tracking error, command timing and Servo-to-WMX latency. Fix timing here, not on hardware. | C | Table R3 filled | |

Exit: a policy episode succeeds through WMX-R2; the controller PR is merged.

### Phase 4: real CR3A (weeks 7 to 9)

Goal: real demonstrations, real training, real evaluation.

| ID | Task | Owner | Done when | Status |
|---|---|---|---|---|
| P4.1 | Hardware: print the T and tip, laminate the sheet, mount the overhead camera. Confirm EtherCAT enabled in `/opt/wmx3/Module.ini`, licence valid, E-stop tested. | C | Checklist signed in `doc/6_real.md` | |
| P4.2 | Enable per-axis soft limits in `cr3a_wmx_parameters.xml` from the Phase 2 joint envelope plus margin (currently disabled on all axes). Verify Servo joint limits and that the gripper closes on the tip. | C | A commanded out-of-limit move is refused | |
| P4.3 | `cameras_real.launch.py`: overhead RealSense at 1280 x 720, 30 fps, on `/image_top/*`; second `apriltag_detector` on it; scorer in real mode. Workspace alignment: jog the tip to three sheet marks, store the sheet-to-base transform. | C | Scorer reports coverage from the tags; tags detected across the whole area | |
| P4.4 | Low-speed path check: Servo `scale.linear` 0.1, guard step 2 mm, Quest teleop with `use_sim_time:=false` everywhere. Verify z lock, clamp, watchdog, dead-man and E-stop. Confirm `/wmx/moveit2/eef_pose` matches the sim frames. | C | Recorder and runner run unchanged against the real robot | |
| P4.5 | Collect 100 to 150 real episodes on the numbered start marks, two lighting conditions, one operator. | C | Dataset `movensys/pusht_cr3a_real_v1` | |
| P4.6 | Train the three policies on real data, 3 seeds. Pick one checkpoint per policy by validation loss before any real evaluation. | M | 9 checkpoints; selection written down before P4.7 | |
| P4.7 | Deploy per checkpoint: shadow, then slow, then normal. 30 trials per policy on the same 30 start marks. Record overhead video, WMX tracking error and timing per trial. | M + C | Table R4 filled | |
| P4.8 | `doc/6_real.md`: safety checklist and procedure. | M + C | Doc complete | |

**Before every autonomous run:** E-stop within reach and tested; soft limits enabled; guard clamp, z lock, watchdog and step limit verified; releasing the dead-man holds the arm; shadow then slow before normal; an operator ready to stop. Never call `/wmx/axis/homing` on the CR3A (it redefines the absolute-encoder zero).

### Phase 5: results (week 10)

| ID | Task | Owner | Done when | Status |
|---|---|---|---|---|
| P5.1 | Regenerate all tables and plots with `compare.py` from the committed CSVs. | M | Section 4 tables match the CSVs | |
| P5.2 | Failure taxonomy per policy: stuck at the edge, overshoot, wrong side of the T, oscillation at chunk boundaries. | M | Counts per category in `doc/7_results.md` | |
| P5.3 | Sim vs real comparison, training cost, latency, smoothness summary; video montage; one-page management summary. | M + C | `doc/7_results.md` complete | |
| P5.4 | Final pass on `doc/1_setup.md` to `doc/6_real.md`. | C | Reproduction by a second engineer | |

Optional, only after M4, in this order: action horizon 16/8 vs 32/16; observation history 1 vs 2; LeRobot real-time chunking on the Flow Matching policy; DiT flow-matching baseline (`multi_task_dit`, 1 seed); wrist camera on or off.

---

## 4. Results tracker

Fill from the CSVs in `results/<stage>/`. Mean and standard deviation over training seeds; real trials reported separately from seed variance. Success = coverage >= 0.95 at any step within 300 steps (10 Hz); 0.90 is the secondary threshold.

| Metric | Definition |
|---|---|
| Success rate | episodes with coverage >= 0.95 (and >= 0.90) |
| Avg max coverage | max coverage per episode, averaged |
| Time to success | steps to threshold, successful episodes only |
| Path length | integrated pusher travel (m) |
| Smoothness | mean absolute jerk of the 50 Hz guarded targets; on HIL and real also RMS joint tracking error |
| Latency | wall-clock per action chunk on the 5090, mean and p95 |
| Training cost | gradient steps, wall-clock, peak VRAM |
| Operator cost | episodes per hour, discard rate |

**R1. gym-pusht** (100 k steps, batch 64, 3 seeds, 50 episodes per checkpoint)

| Policy / variant | Success 0.95 | Success 0.90 | Avg max cov | Train h | VRAM GB | Latency ms (p95) |
|---|---|---|---|---|---|---|
| Diffusion, 200 k reference (model card: 65.4 %) | 58 (64 at step 200k in-training eval) | 78 | 0.87 | 4.5 | 5.0 | not measured |
| ACT default (chunk 100, ensembling) | | | | | | |
| ACT matched 16/8 | | | | | | |
| ACT matched, no VAE | | | | | | |
| Diffusion DDIM 10 | | | | | | |
| Diffusion DDPM 100 | | | | | | |
| Flow Matching Euler 1 / 2 / 4 / 8 / 16 | | | | | | |

**R2. Isaac Sim** (50 start states, 3 seeds)

| Policy | Success 0.95 | Success 0.90 | Avg max cov | Time to success | Path length | Smoothness | Latency ms |
|---|---|---|---|---|---|---|---|
| ACT | | | | | | | |
| Diffusion | | | | | | | |
| Flow Matching (K = ) | | | | | | | |

Dataset: episodes collected ___, kept ___, discard rate ___, episodes per hour ___.

**R3. Hardware-in-the-loop** (10 teleop + 10 policy episodes)

| Controller | RMS tracking error (rad) | Command period jitter (ms) | Servo-to-WMX latency (ms) | Policy episode success |
|---|---|---|---|---|
| | | | | |

**R4. Real CR3A** (30 trials per policy, one checkpoint each, 95 % confidence interval about +/- 17 points at 30 trials)

| Policy | Success 0.95 | Success 0.90 | Avg max cov | Time to success | Smoothness | Tracking error | Latency ms |
|---|---|---|---|---|---|---|---|
| ACT | | | | | | | |
| Diffusion | | | | | | | |
| Flow Matching (K = ) | | | | | | | |

Dataset: episodes collected ___, kept ___, lighting conditions ___, operator ___.

---

## 5. System design (what each phase builds against)

### Task

| Parameter | Value (all in `config/dobot_cr3a/task.yaml`) |
|---|---|
| Robot | Dobot CR3A, `joint1..joint6`, EEF `Link6`, planning frame `world_manipulator` |
| Pusher | Printed 20 mm cylinder tip held by the closed gripper; tool axis vertical, fixed yaw, tip bottom 10 mm above the table; only x, y move |
| Push area | 0.45 x 0.45 m square in front of the robot, centre fixed after the P2.1 reach check |
| T block | gym-pusht proportions at **40 mm per unit**: bar 160 x 40 mm, stem 40 x 120 mm, height 30 mm, 200 to 300 g; AprilTag 36h11 id 10 (36 mm) at the bar-stem junction, same in sim and real |
| Target | T outline printed on a laminated sheet with AprilTag id 20 (50 mm) at a known offset and 30 numbered start marks with orientation |
| Start states | T position uniform in the inner 0.30 x 0.30 m, yaw uniform; EEF home at one corner. Seeds 0 to 49 evaluate, 1000+ collect; 30 marks on the real sheet |
| Rates | policy 10 Hz, guard and teleop 50 Hz, Servo 40 Hz, WMX3 1 kHz |
| Episode | 300 policy steps (30 s) |

### Dataset schema (same for gym conversion, sim and real)

| Key | Shape | Content | Sim source | Real source |
|---|---|---|---|---|
| `observation.images.top` | video 256 x 256 x 3 | overhead RGB cropped to the push area, crop anchored on the sheet tag | `/image_top/rgb` | `/image_top/rgb` |
| `observation.state` | float32 (2,) | pusher x, y in `world_manipulator` (m) | `/wmx/moveit2/eef_pose` | same |
| `action` | float32 (2,) | guarded target x, y at this 10 Hz tick | `/pusht/target_guarded` | same |
| `observation.joints` | float32 (6,) | joint positions (analysis only) | `/joint_states` | same |
| `block_pose` | float32 (3,) | T x, y, yaw in the sheet frame (not a policy input) | `/pusht/t_block/pose` | tag 10 relative to tag 20 |
| `coverage` | float32 (1,) | scorer output | scorer | scorer |

Episode metadata: start-state id, operator, sim or real, Servo scale, guard parameters, software versions, success, time to success. Images are decoded from `sensor_msgs/Image` with numpy, never `cv_bridge` (apt `cv_bridge` is built against numpy 1.26; LeRobot needs numpy 2).

### Control path

```
policy_runner (10 Hz chunks)  ─┐
                               ├─► planar_guard ──50 Hz──► /servo_node/pose_target_cmds ──► servo_node (POSE, 40 Hz)
quest_servo_teleop (50 Hz)    ─┘   (host venv)                                                     │ JointTrajectory
                                                               ┌───────────────────────────────────┼──────────────────────┐
                                                        SIM: sim_bridge ► Isaac      HIL: wmx-r2 ► WMX simu ► Isaac 9b     REAL: wmx-r2 ► EtherCAT ► CR3A
```

| Process | Where |
|---|---|
| Isaac Sim (scene 9a or 9b) | host, system Jazzy sourced |
| wmx-r2 nodes + streaming controller | host, `sudo --preserve-env` (real-time scheduling, licence) |
| `move_group`, `servo_node`, `sim_bridge`, `moveit2_api`, `quest_servo_teleop`, RealSense, AprilTag | manipulator container (`isaac-ros_4.1`, Jazzy) |
| `quest_pose_publisher` + CloudXR | teleop container |
| guard, recorder, episode manager, scorer, policy runner | host venv; standard messages only, so nothing from the containers is built on the host |

`ROS_DOMAIN_ID=70` everywhere (containers inherit it from the host shell; ignore the 73 in the manipulator README). `use_sim_time` true in sim and HIL, false on the real robot; the real launch file overrides the teleop defaults.

**planar_guard** is the one safety and consistency layer. It takes targets from teleop (`/pusht/target_raw`) or from the runner (timed 10 Hz waypoint lists, interpolated to 50 Hz), selected by mode so there is never more than one writer. It overwrites z and orientation, clamps x, y to the push area, limits the step (5 mm per cycle) and the distance from the measured EEF (30 mm, otherwise hold), holds on stale input (0.5 s) or released dead-man, stops publishing on fault so Servo halts, keeps Servo in POSE mode, and publishes `/pusht/target_guarded` (recorded as `action`).

**policy_runner** builds the observation at 10 Hz, calls `predict_action_chunk`, hands the next `n_action_steps` targets to the guard and starts the next inference while the chunk executes. Modes: `shadow` (log only, human teleoperates), `slow` (Servo scale 0.1, step 2 mm), `normal`. Safety layers on the real arm, outside in: E-stop and dead-man; WMX soft limits; Servo joint limits and self-collision checking (the table is not a collision object, because Servo's 20 mm proximity threshold would stall a tip 10 mm above it); the guard; shadow then slow then normal for every new checkpoint.

**Episode loop** (`episode_manager`): reset (sim: teleport the T, home the EEF through the guard; real: show the start mark, wait for the operator), record at exactly 10 Hz, stop on coverage >= 0.95, 300 steps or a key, score, save or discard. Keyboard plus a `Trigger` service for scripted sweeps.

**Real-robot sensing.** Overhead RealSense at 1280 x 720, 30 fps (a 36 mm tag at 0.7 m is about 48 px, enough for tag36h11; at 640 x 480 it would be about 24 px and unreliable). The scorer reads the two tag TFs and computes the T pose in the sheet frame, so no camera-to-robot calibration is needed for scoring. The wrist will occlude the T tag during pushes, so the scorer keeps the last valid pose and always scores once more after the arm retracts at episode end. The sheet-to-base transform from the three-mark alignment maps start marks to base-frame poses and is not used by the policy.

### Repository layout

```
movensys-imitation/
├── movensys_imitation/          policies/flow_matching_unet/  tasks/pusht/  data/  configs/  cli/  compare.py  tests/
├── movensys_imitation_ros/      planar_guard  episode_recorder  episode_manager  pusht_scorer  policy_runner  eval_sim  eval_real
│   ├── launch/                  collect.launch.py  run_policy.launch.py  cameras_real.launch.py   (mode:=sim|hil|real)
│   └── config/dobot_cr3a/       task.yaml  guard.yaml  recorder.yaml  real_calibration.yaml  apriltag_top.yaml
├── sim/isaacsim/                9a_pusht_simulation.usd  9b_pusht_hil.usd  assets/
├── hardware/                    t_block.stl  pusher_tip.stl  target_sheet.pdf  camera_mount.md
├── results/                     committed CSVs per stage (tables regenerate from these)
├── doc/                         1_setup  2_task  3_gym  4_sim  5_hil  6_real  7_results
└── plan/                        this file
```

`datasets/` and `outputs/` are ignored. Dataset ids: `movensys/pusht_gym`, `movensys/pusht_cr3a_sim_v1`, `movensys/pusht_cr3a_real_v1`.

---

## 6. Policies and fairness protocol

| | ACT | Diffusion Policy | Flow Matching (`flow_matching_unet`, ours) |
|---|---|---|---|
| Encoder | ResNet18 + transformer | ResNet18 + spatial softmax | identical to Diffusion (imported, not copied) |
| Input | 96 x 96 | 84 x 84 random crop of 96 (centre crop at eval) | same as Diffusion |
| `n_obs_steps` | 1 | 2 | 2 |
| Horizon / executed | default chunk 100 with temporal ensembling; matched 16 / 8 | 16 / 8 | 16 / 8 |
| Objective | L1 + VAE KL (also a no-VAE run) | DDPM epsilon, 100 train steps; DDIM 10 at inference | OT path `x_t = (1 - t) noise + t a`, target `a - noise`, MSE; Euler K steps, `t` uniform or Beta(1.5, 1) |
| Normalisation | MIN_MAX state and action, MEAN_STD images (same for all) | same | same |
| Budget | 100 k steps, batch 64, 3 seeds | same | same |

Flow Matching lives in `movensys_imitation/policies/flow_matching_unet/` as `configuration_`, `modeling_` and `processor_` files, registered as `flow_matching_unet` through LeRobot's config registry, reusing `DiffusionRgbEncoder` and `DiffusionConditionalUnet1d`. Config adds `fm_num_steps`, `fm_solver`, `fm_time_sampling`.

Fairness: same dataset and split, same resolution and augmentation where the method allows, same normalisation, same horizon where the method allows plus the method's own default, same step and batch budget, three seeds, fixed start states, training time, VRAM and latency measured on the same GPU. Hyperparameters freeze after Phase 1. ACT is expected to score low on PushT (L1 regression averages a multimodal distribution); that is a finding, not a bug. ACT at batch 64 departs from LeRobot's default of 8, so its learning rate may be adjusted once in Phase 1 and recorded.

---

## 7. Risks

| Risk | L / I | Mitigation |
|---|---|---|
| Streaming controller not merged or jerky on six axes | M / H | Decide in P0.6; HIL (Phase 3) before hardware; report the controller with the smoothness numbers |
| Quest / CloudXR bring-up slow | M / M | Start week 1; fallbacks: `keyboard_teleop` for pilots, a mouse-on-image teleop node if the Quest is unavailable |
| Demonstrations through Servo too slow or jerky | M / H | Pilot and tuning (P2.5), replay check, keep successes only |
| Tags too small or occluded on the real setup | M / M | 1280 x 720, 36 mm tag on a 40 mm per unit T, end-of-episode scoring; fallback: colour segmentation of the T |
| Sim-to-real image gap | H / M | Real policies train on real data; sim validates the pipeline; same tag-anchored crop; two lighting conditions |
| Schedule has no slack | H / M | Phase 0 is two weeks; cut real-data seeds to 2 before cutting phases |
| Real-robot damage from a bad action | L / H | Phase 4 checklist and the five safety layers |
| numpy 2 breaks system Python packages in the venv | H / L | Only `rclpy`, `tf2_ros` and message packages from the system; numpy image decoding |
| Isaac Sim 5.1 is a release candidate | L / M | Scenes are plain USD + OmniGraph |

---

## 8. Decisions needed before week 2, and assumptions

1. **Robot PC.** Is this workstation also the robot PC (WMX3 licence and EtherCAT NIC)? If not, Phase 4 moves wmx-r2 and the runner there.
2. **Streaming controller.** Who owns P0.6 and which controller is the first choice.
3. **Overhead camera.** Re-mount the existing nvblox RealSense or buy one more D435 (about KRW 0.5 M).
4. **Scene ownership.** Contribute `9a`/`9b` to `movensys-simulation` (recommended) or keep them here.
5. **Accept the v2 changes** listed at the top (T scale, camera resolution, seed rule, 30 trials, two-week Phase 0).

Assumptions: the Quest 2 and the person who did the VR bring-up are available; the T can be pushed at 10 mm tip height without the fingers touching the table; 0.95 coverage is reachable with the printed T (otherwise 0.90 becomes primary and is reported as such); one operator per dataset.

---

## 9. Project review

At each review answer only these five questions, then update the master tracker and the results tables:

1. What was completed since the last review?
2. What is the next task?
3. What is blocking?
4. What evidence shows the completed work works (link in the tracker)?
5. Is the next milestone still achievable?

Detailed topics, commands, branch notes and safety procedures live in `doc/`, not here.

---

## Appendix A. Environment and key interfaces

**Venv** (uv on system Python 3.12 with `--system-site-packages` so Jazzy's `rclpy` is importable next to LeRobot):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source /opt/ros/jazzy/setup.bash && cd ~/workspaces/movensys-imitation
uv venv --python 3.12 --system-site-packages .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128      # RTX 5090 needs cu128
uv pip install -e ".[dev,viz]"      # lerobot[pusht,diffusers-dep,dataset,training]==0.6.1, scipy, matplotlib, shapely, pygame, pytest, ruff
python -c "import torch, lerobot, rclpy, numpy; print(torch.cuda.is_available(), lerobot.__version__, numpy.__version__)"
```

LeRobot 0.6.1 needs Python >= 3.12, torch 2.7 to 2.11, numpy 2.0 to 2.2. Isaac Sim's own Python is 3.11, so no Isaac-side Python is used; reset and pose publishing are OmniGraph nodes.

**Interfaces that the new nodes depend on:**

| Interface | Type | Owner | Notes |
|---|---|---|---|
| `/quest_pose_publisher/controller_pose_right`, `/joy_right` | `PoseStamped`, `Joy` | teleop container | about 55 Hz; squeeze >= 0.6 is the clutch and the dead-man |
| `/pusht/target_raw` | `PoseStamped` | `quest_servo_teleop` (remapped) | 50 Hz while clutched |
| `/servo_node/pose_target_cmds` | `PoseStamped` (`world_manipulator`) | `planar_guard` | 50 Hz, always |
| `/servo_node/switch_command_type` | `moveit_msgs/srv/ServoCommandType` | guard calls it | 2 = POSE |
| `/movensys_manipulator_arm_controller/joint_trajectory` | `JointTrajectory` | `servo_node`, 40 Hz | consumed by `sim_bridge` or the wmx-r2 streaming controller |
| `/joint_command`, `/joint_states` | `JointState` | Isaac (sim), `/isaacsim/...` in HIL; wmx-r2 at 100 Hz on real | |
| `/wmx/moveit2/eef_pose` | `PoseStamped` | `moveit2_api` (devel), 100 Hz | `observation.state` |
| `/wmx/set_gripper` | `std_srvs/SetBool` | `sim_bridge` (sim), `gripper_controller` (real) | true = close on the tip |
| `/image_top/rgb`, `/image_top/camera_info` | `Image`, `CameraInfo` | Isaac camera or RealSense | 1280 x 720, 30 Hz |
| `/pusht/t_block/pose`, `/pusht/t_block/pose_cmd` | `geometry_msgs/Pose` | Isaac graphs | ground truth 30 Hz; teleport on reset |
| TF `camera_top_color_optical_frame -> tag36h11:10`, `:20` | TF | `apriltag_detector` (real) | T and sheet tags |
| `/wmx/engine/ready`, `/wmx/core_motion/ready` | `Bool` (transient local) | wmx-r2 | wait for both before streaming |

Frames: `world_manipulator` is the planning frame (`Link0` at (-0.03, 0.275, 0.05), yaw -90 deg); `Link6` is Servo's `ee_frame`; table top at z about 0.043 in sim.

## Appendix B. References

LeRobot 0.6.1 (policy factory, `PreTrainedPolicy.predict_action_chunk`, `lerobot/diffusion_pusht` model card: 65.4 % success, 0.95 threshold); gym-pusht; Chi et al. 2023 (Diffusion Policy); Zhao et al. 2023 (ACT); Lipman et al. 2023 (Flow Matching); Black et al. 2024 (pi0) and 2025 (Real-Time Chunking); Isaac Sim 5.1 ROS 2 notes; internal VR teleop notes and the Physical AI Studio report.
