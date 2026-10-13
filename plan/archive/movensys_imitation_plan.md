# movensys-imitation: implementation plan

ACT, Diffusion Policy and Flow Matching compared on PushT with the Dobot CR3A. Demonstrations are collected with the Meta Quest teleop stack through MoveIt Servo, commands are executed through WMX-R2, first in Isaac Sim and then on the real arm.

| | |
|---|---|
| Status | Draft v1.1 for team review (v1.1: movensys-teleoperation checkout switched to `devel`, domain-ID handling corrected) |
| Host this plan was checked against | Ubuntu 24.04.3, ROS 2 Jazzy, Python 3.12.3, RMW CycloneDDS, RTX 5090 32 GB (driver 590), 32 CPU threads, 31 GB RAM, 201 GB free disk, Isaac Sim 5.1.0-rc.19 in `~/isaac-sim`, WMX3 SDK in `/opt/wmx3` |
| Supersedes | `Sync/Movensys_workspace/movensys-imitation/doc/plan.md` (earlier draft written before the sibling repos and this host were checked; several of its assumptions were wrong, see Section 2) |

---

## 0. Summary

We build one repository, `movensys-imitation`, that trains and runs three imitation-learning policies (ACT, Diffusion Policy, Flow Matching) on one task, PushT (push a T-shaped block onto a target outline), with one data format (LeRobot dataset), one evaluation harness and one control path. Demonstrations come from the Quest controller through `quest_servo_teleop` and MoveIt Servo; the resulting joint targets reach the arm through WMX-R2 on the real robot and through the existing sim bridge in Isaac Sim.

The work runs in four stages that each reuse the previous stage's code:

1. **gym-pusht** (pure Python): validates the three policies and our Flow Matching implementation against published numbers.
2. **Isaac Sim with the CR3A**: validates the whole ROS 2 pipeline (teleop, recorder, policy runner, scoring) with ground-truth block poses.
3. **Hardware-in-the-loop**: the WMX3 simulation platform drives Isaac Sim through WMX-R2, which validates the real streaming path without touching hardware.
4. **Real CR3A**: real demonstrations, real evaluation, 20 trials per policy.

The deliverable is a comparison table (success rate, coverage, time to success, smoothness, inference latency, training cost) for the three policies in simulation and on the real arm, plus a pipeline that can be pointed at the next task.

Effort: about 10 weeks for one engineer full time, 7 to 8 weeks with a controls engineer and an ML engineer working in parallel (Section 8).

---

## 1. Goal, scope and done criteria

**Goal.** Measure, on the same task and the same robot, how ACT, Diffusion Policy and Flow Matching differ in success rate, motion quality and inference cost, and ship the tooling that makes such comparisons repeatable at Movensys.

**Deliverables.**

1. `movensys-imitation` repository (MIT, same conventions as the sibling repos): a Python library (policies, task geometry, scoring, dataset tools), a ROS 2 package (teleop guard, episode recorder, episode manager, policy runner, scorer), Isaac Sim scene assets, printable hardware files and documentation.
2. A Flow Matching policy implemented as a LeRobot policy that shares the Diffusion Policy encoder and UNet so that the only variable is the generative objective.
3. Three datasets: gym-pusht (public), PushT-CR3A-sim, PushT-CR3A-real.
4. Trained checkpoints for the three policies (three seeds each) on each dataset.
5. A results document with the comparison tables, videos and a failure taxonomy.

**In scope.** PushT only. Policies trained from scratch on our data. Quest teleop as the demonstration source. Isaac Sim as the simulator. WMX-R2 as the real-robot execution layer.

**Out of scope (for this plan).** Fine-tuning pretrained VLAs (pi0, SmolVLA) as a main line (noted as an optional extra), reinforcement learning, other tasks, Jetson deployment, any new WMX3 features.

**Done when.**

- All three policies train and evaluate end to end on gym-pusht with results in the expected range.
- All three policies push the T to the target in Isaac Sim with a measured success rate on a fixed set of 50 start states.
- All three policies run on the real CR3A through WMX-R2 with a measured success rate on 20 trials each.
- The results document and the setup documents are complete enough that a second engineer can reproduce a run.

---

## 2. Starting point (verified on 2026-10-13)

### 2.1 Host

| Item | Value |
|---|---|
| OS, Python | Ubuntu 24.04.3, system Python 3.12.3 (externally managed, so everything goes into a venv) |
| ROS 2 | Jazzy only (`/opt/ros/jazzy`), `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `ROS_DOMAIN_ID=70` in `~/.bashrc` |
| Env vars in `~/.bashrc` | `MOVENSYS_ROS_VERSION=isaac-ros_4.1`, `CPU_ARCH=amd64`, `MANIPULATOR_MODEL=dobot_cr3a`, `MOVENSYS_MANIPULATOR_PACKAGES`, `MOVENSYS_TELEOPERATION_PACKAGES`, `ISAAC_ROS_WS` |
| GPU | RTX 5090 32 GB (Blackwell, sm_120: needs torch >= 2.7 with cu128 wheels) |
| Isaac Sim | 5.1.0-rc.19 in `~/isaac-sim` (the movensys-simulation README says `~/isaacsim`). ROS 2 bridge `isaacsim.ros2.bridge` 4.12.4 is installed with Jazzy libraries including CycloneDDS. |
| WMX3 | `/opt/wmx3`, C++ SDK only, no Python bindings. `Module.ini` currently has EtherCAT disabled and the `simu` platform enabled. `simu_network.def` defines axes 0 to 2 only. |
| Docker | 29.2.1 installed. No images or containers exist yet for the manipulator or teleop stacks. |
| Python tooling | No `uv`, no venvs, no LeRobot or torch installed anywhere. System numpy is 1.26.4 (the apt `cv_bridge` is built against it). |
| ROS workspace | `~/workspaces/movensys_ws` has only `wmx_r2_message`, `wmx_r2_package`, `wmx_r2_control` built. |

### 2.2 Sibling repositories and which branch has what

The code this project depends on is mostly on `devel` branches, not on the checked-out `main` branches.

| Repository (local path) | Checked out | What we need | Where it is |
|---|---|---|---|
| `movensys_ws/src/wmx-r2` | `main` @ b419148 | A subscriber that streams MoveIt Servo's `JointTrajectory` output to the WMX axes | **Not on `main`.** `main` only has a custom `FollowJointTrajectory` action server (`StartCSplinePos`, no streaming). `joint_position_controller` (one `StartLinearIntplPos` per message) and `servo_stream_controller` (WMX API buffer) are on `origin/feature/api-buffer-command-streaming` and `...-with-wait`; a cyclic-buffer `position` command interface with a stock ros2_control JTC at 100 Hz is on `origin/feat/cyclic-buffer-stream-timing`. The VR-teleop notes in `national_project_vision_based_teaching/VR_based_motion_Real` say VR teleop ran through "api buffer or joint position controller", so at least one has been exercised. |
| `movensys_ws/src/movensys-manipulator` | `main` @ f4e7e32 | MoveIt Servo (POSE mode), `quest_servo_teleop`, `sim_bridge`, `moveit2_api` (EEF pose at 100 Hz), RealSense and AprilTag launches, `planar` teleop mode | `origin/devel` @ e242044 (2026-10-13). On `main` the bridge is `isaacsim_bridge`, the API is `trajectory_api`, and `quest_servo_teleop` is the old 333-line version without DOF modes. |
| `movensys_ws/src/movensys-teleoperation` | `devel` @ 50f91f7 (switched from the stub `main` on 2026-10-13) | `quest_pose_publisher` (Quest controller pose and buttons via CloudXR) | Present in the checkout: package, launch file, docker files and `doc/1_setup.md` to `doc/3_quest_pose.md`. Not built yet and the teleop container does not exist yet. Needs the CloudXR runtime from NGC (`~/.cloudxr`, `~/.ngc` do not exist yet) and the `isaac-ros_4.1` teleop container. |
| `workspaces/movensys-simulation` | `main` @ 3550b15 | CR3A Isaac scenes with `/joint_command`, `/joint_states`, `/clock`; an overhead camera; pose publish/subscribe graphs for objects | `dobot_cr3a/3a_trajectory_simulation.usd` is the clean base. The RealSense top-camera model exists only in `7b_robopoly_hil.usd` (at (0.35, 0, 0.7) looking down); in `7a`, `8a`, `8b` the top camera is an empty placeholder. Object pose publishers and pose-command subscribers exist in the `7x` scenes. No T block, no TF publisher, no standalone Python app. |
| `Sync/.../intel_collaboration/physical AI Studio` | report only | Prior experience: ACT (35k and 100k steps) and SmolVLA trained on an 85-episode cube pick-and-place dataset with Intel's Physical AI Studio (pins `lerobot[dataset]==0.6.0`, `gym-pusht`) | Reference for dataset conventions and a second PushT gym wrapper (`physicalai/gyms/pusht.py`). |

### 2.3 LeRobot 0.6.1 (PyPI, released 2026-08-03)

- `requires_python >= 3.12`, `torch >= 2.7, < 2.12`, `torchvision >= 0.22, < 0.27`, `numpy >= 2.0, < 2.3`, `gymnasium >= 1.1.1, < 2`.
- `lerobot[pusht]` pulls `gym-pusht >= 0.1.5, < 0.2` and `pymunk >= 6.6, < 7`; `lerobot[diffusers-dep]` pulls `diffusers >= 0.38, < 0.40` (needed by Diffusion Policy); `lerobot[dataset]` pulls `torchcodec`.
- Built-in policies include `act`, `diffusion`, `multi_task_dit` (a DiT with `objective` = `diffusion` or `flow_matching`, CLIP ViT-B/16 image and text encoders, 224 px crops), `rtc` (real-time chunking inference wrapper), `pi0`, `pi05`, `smolvla`, `vqbet`, `tdmpc`. There is no ResNet-UNet flow-matching policy; we implement that.
- Policy types resolve through the config registry (`@PreTrainedConfig.register_subclass("<type>")`); `get_policy_class` imports `modeling_<type>` next to `configuration_<type>` and expects the class name `<ConfigName minus Config>Policy`. An external package can therefore add a policy type without patching LeRobot, as long as it is imported before the CLI parses `--policy.type`.
- `PreTrainedPolicy` exposes `select_action` (one action, internal queue of `n_action_steps`) and `predict_action_chunk` (the whole chunk), which the policy runner uses for interpolation.
- Reference numbers on gym-pusht from the `lerobot/diffusion_pusht` model card: success 65.4 % over 500 episodes (success = max overlap >= 0.95), average max overlap 0.955; the original Diffusion Policy code scores 64.2 % / 0.957 under the same protocol; the keypoint (state) variant reaches 71.0 %. Community ACT checkpoints on PushT report near-zero success, which is consistent with L1 regression averaging a multimodal pushing distribution. We expect ACT to trail on this task and we report it as a finding, not a bug.

### 2.4 Isaac Sim 5.1

- Supported pairing is Ubuntu 24.04 with ROS 2 Jazzy. `isaac-sim.sh` sources `setup_ros_env.sh`, which adds the bundled Jazzy bridge libraries when `ROS_DISTRO` is unset; with system Jazzy sourced (the movensys-simulation README workflow) the system libraries are used, which is fine because the bridge is OmniGraph, not rclpy.
- Isaac Sim's own interpreter is Python 3.11, so a standalone `python.sh` app cannot import the system (3.12) `rclpy`. This plan avoids Isaac-side Python entirely: reset and block-pose publishing use OmniGraph ROS 2 nodes, the same pattern the Monopoly scenes already use. A standalone app remains an option for headless batch evaluation later.
- Camera prims default to 1280x720 and publish every render frame (60 Hz) unless resolution and frame skipping are set.

### 2.5 Gap list: what exists and what must be built

| Need | Exists today | Gap we close |
|---|---|---|
| Quest controller to Servo POSE targets | `quest_pose_publisher` + `quest_servo_teleop` (devel), 50 Hz, clutch on grip squeeze, `planar` mode (XY + yaw), `custom_dof_gain` for XY only | Teleop repo is on `devel` already; switch the manipulator repo to `devel` too, build both containers, add CloudXR. Masking in the teleop node is "not a hard constraint", so a **planar guard** node enforces fixed z and orientation (Section 3.4). |
| Servo to Isaac Sim | `sim_bridge` (devel): Servo `JointTrajectory` to `/joint_command`, `/wmx/set_gripper` served in sim | None. |
| Servo to real CR3A through WMX-R2 | Only on wmx-r2 remote branches (see 2.2) | Choose one controller, merge to `main`, verify on the WMX simu platform (HIL), then on hardware (Phase 0 and 2b). |
| CR3A Isaac scene for PushT | `3a_trajectory_simulation.usd` (robot, joint graphs, clock), top camera model in `7b`, object pose graphs in `7x` | New scene `9a_pusht_simulation.usd` and `9b_pusht_hil.usd`: T block, target sheet, pusher tip, overhead camera at 30 Hz, T pose publisher, T pose-command subscriber. |
| Overhead camera on the real setup | RealSense "nvblox" camera (serial 234222302502) at 640x480, 15 fps; hand camera at 6 fps | Mount the nvblox camera above the push area, raise to 30 fps, `/image_top/*` naming to match the sim. |
| Block pose on the real setup | `apriltag_detector.py` (tag36h11, 50 mm, hand camera, TF output) | Second detector instance on the overhead image; tags on the T and on the target sheet; scorer computes the T pose relative to the sheet tag (no camera-to-base calibration needed for scoring). |
| EEF pose for `observation.state` | `/wmx/moveit2/eef_pose` (PoseStamped, 100 Hz, devel) and TF `world_manipulator -> Link6` | None. |
| Gripper (holds the pusher tip) | `/wmx/set_gripper` (`std_srvs/SetBool`), served by `sim_bridge` in sim and `gripper_controller` on hardware | The hardware node hard-codes `CreateDevice("/opt/lmx/")`, which does not exist here; fix in wmx-r2. |
| Dataset recorder, episode manager, scorer, policy runner | Nothing (only `ros2 bag record` of one image topic and a JPEG capture script) | Build (Section 3.4, 3.5, 4). |
| Policies | ACT and Diffusion in LeRobot; DiT with flow matching in LeRobot | Our ResNet-UNet Flow Matching policy; thin train/eval entry points that register it. |
| Python environment | None | uv-managed venv on system Python 3.12 with `--system-site-packages` for `rclpy` (Section 3.6). |
| WMX safety configuration | `cr3a_wmx_parameters.xml` has soft limits disabled on all axes, `InPosWidth` 1000 | Enable per-axis soft limits that cover the push workspace with margin before any real run. |

---

## 3. System design

### 3.1 Task: PushT on the CR3A

The task follows gym-pusht so that stage 1 numbers transfer: a planar pusher moves a T-shaped block into a T-shaped target region; the policy observes an overhead image and the pusher position and outputs the pusher's next target position.

| Parameter | Value (initial; all live in `config/dobot_cr3a/task.yaml`) |
|---|---|
| Robot | Dobot CR3A, joints `joint1..joint6`, EEF link `Link6`, planning frame `world_manipulator` |
| Pusher | Printed cylindrical tip, diameter 20 mm, held by the closed gripper (`/wmx/set_gripper true`), identical geometry in sim (cylinder prim under `Link6`) and real |
| Tool pose | Tool axis vertical (pointing down), fixed yaw, fixed z such that the tip bottom is 10 mm above the table surface; only x and y move |
| Push area | 0.45 x 0.45 m square on the table in front of the robot; the exact centre is fixed in Phase 2 after a reach check in sim |
| T block | gym-pusht proportions at 25 mm per unit: bar 100 x 25 mm, stem 25 x 75 mm, height 30 mm, mass 150 to 200 g (solid PLA or added weight), AprilTag 36h11 id 10 (40 mm) on the bar top in sim and real |
| Target | T outline printed on a paper sheet taped to the table, with AprilTag 36h11 id 20 (50 mm) at a known offset; the sheet also carries the numbered start marks used for reproducible real-world episodes |
| Start states | T position uniform inside the inner 0.30 x 0.30 m, yaw uniform; EEF starts at a fixed home position at one corner of the area. Seeds 0 to 49 are the evaluation set, 1000+ the collection set. |
| Control rate | Policy 10 Hz (gym-pusht convention), teleop and runner stream Servo targets at 50 Hz, Servo publishes at 40 Hz, WMX3 at 1 kHz |
| Episode limit | 300 policy steps (30 s) |
| Success | Coverage = area(T footprint intersect target) / area(T) >= 0.95 at any time in the episode (gym-pusht's threshold); 0.90 reported as a secondary metric |

### 3.2 Observations, actions and the dataset schema

The same schema is used for gym-pusht conversion, sim and real, so a checkpoint trained on one can be evaluated on another.

| Key | Shape, dtype | Content | Source in sim | Source in real |
|---|---|---|---|---|
| `observation.images.top` | video, 256 x 256 x 3 | Overhead RGB cropped to the push area (square crop anchored on the target-sheet tag, fallback fixed pixel crop in config). Policies resize to 96 x 96 (ACT, Diffusion, Flow Matching) or 224 (DiT-FM). | Isaac camera `/image_top/rgb` | RealSense `/image_top/rgb` |
| `observation.state` | float32 (2,) | Pusher x, y in `world_manipulator` (metres) | `/wmx/moveit2/eef_pose` | same |
| `action` | float32 (2,) | Commanded pusher target x, y in `world_manipulator` at this 10 Hz tick (the guard's output) | `/pusht/target_guarded` | same |
| `observation.joints` (extra) | float32 (6,) | Joint positions, for analysis and for joint-space policies later | `/joint_states` | same |
| `block_pose` (extra, not a policy input) | float32 (3,) | T x, y, yaw relative to the target sheet frame | Isaac pose publisher `/pusht/t_block/pose` | AprilTag TF `tag36h11:10` relative to `tag36h11:20` |
| `coverage` (extra) | float32 (1,) | Scorer output per frame | scorer | scorer |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | LeRobot standard | | | |

Episode metadata (JSON sidecar and dataset `info`): start-state id, operator, sim or real, Servo scale, guard parameters, software versions, success flag, time to success.

Images are stored as video at 256 x 256 (LeRobot encodes with ffmpeg); the venv nodes decode `sensor_msgs/Image` with numpy (`frombuffer` + `reshape`), never with `cv_bridge`, because the apt `cv_bridge` is linked against numpy 1.26 while LeRobot requires numpy 2.

### 3.3 Control path and process placement

```
                 10 Hz action chunk (x, y)            50 Hz PoseStamped
 policy_runner ───────────────────────► planar_guard ──────────────────► /servo_node/pose_target_cmds
 (host venv)      or                    (host venv)                       │
 quest_servo_teleop ───► /pusht/target_raw ─┘                             ▼
 (manipulator container, 50 Hz)                                  servo_node, POSE mode, 40 Hz
                                                                 (manipulator container)
                                                                          │ /movensys_manipulator_arm_controller/joint_trajectory
                                   ┌──────────────────────────────────────┼─────────────────────────────────────┐
                                   ▼                                      ▼                                     ▼
                      SIM: sim_bridge (container)           HIL: wmx-r2 streaming controller        REAL: wmx-r2 streaming controller
                      /joint_command ──► Isaac Sim          WMX3 simu platform ──► /isaacsim/        WMX3 ──► EtherCAT 1 kHz ──► CR3A
                      /joint_states ◄── Isaac Sim           joint_command ──► Isaac Sim (9b scene)   /joint_states ◄── WMX3 (100 Hz)
```

| Process | Runs where | Notes |
|---|---|---|
| Isaac Sim 5.1 (GUI, scene `9a`/`9b`) | host | System Jazzy sourced, `ROS_DOMAIN_ID` aligned (see below) |
| wmx-r2 nodes (`wmx_r2_cr3a_manipulator.launch.py` + streaming controller) | host, `sudo --preserve-env` | Root is needed for real-time scheduling and the license |
| MoveIt (`move_group`, `servo_node`), `sim_bridge`, `moveit2_api`, `quest_servo_teleop`, RealSense and AprilTag nodes | `movensys_manipulator_container` (track `isaac-ros_4.1`, Jazzy) | The `isaac-ros_4.1` Dockerfile does not install `ros-jazzy-moveit-servo` (the `general` one does); add it |
| `quest_pose_publisher` + CloudXR | `movensys_teleoperation_container` (Jazzy) | Quest connects over Wi-Fi through the WebXR client on port 48322 |
| `planar_guard`, `episode_recorder`, `episode_manager`, `pusht_scorer`, `policy_runner` | host venv (`movensys-imitation/.venv`) | Standard messages only (`geometry_msgs`, `sensor_msgs`, `std_srvs`, `trajectory_msgs`, `moveit_msgs` for the Servo command-type switch, `tf2_ros`); no custom srv dependency so nothing from the containers has to be built on the host |

**Domain ID.** `~/.bashrc` and the movensys-simulation README use `ROS_DOMAIN_ID=70`; the manipulator README says 73. Both the manipulator and the teleop compose files pass `${ROS_DOMAIN_ID}` through from the host environment, so everything in this project uses 70 by starting the containers from a shell that has the `~/.bashrc` value, and `env.bash` in this repo sets the same value for the venv nodes.

**Sim time.** In sim and HIL every node runs with `use_sim_time:=true` (Isaac publishes `/clock`). On the real robot it must be false everywhere; `quest_servo_teleop.launch.py` defaults to true and `keyboard_teleop` hard-codes true, so the real-robot launch file in this repo overrides them. `quest_pose_publisher` and `quest_servo_teleop` must agree, otherwise the 0.15 s pose watchdog drops the clutch.

### 3.4 Demonstration collection (Quest teleop through Servo)

1. `quest_pose_publisher` publishes the right controller pose (`/quest_pose_publisher/controller_pose_right`) and buttons (`/quest_pose_publisher/joy_right`, axes `[thumb_x, thumb_y, trigger, squeeze]`, buttons `[primary, secondary, thumbstick, menu]`) at about 55 Hz.
2. `quest_servo_teleop` (devel) turns that into relative `PoseStamped` targets at 50 Hz while the grip is squeezed (clutch), with `motion_mode: planar` and `custom_dof_gain: [1, 1, 0, 0, 0, 0]` (XY only), `position_scale` 1.0 for 1:1 hand-to-table mapping (0.5 if finer control is needed), `max_target_step` 0.05 m. Its output topic is remapped from `/servo_node/pose_target_cmds` to `/pusht/target_raw`.
3. **`planar_guard`** (new, this repo) is the single safety and consistency layer for both teleop and policy input. It subscribes to `/pusht/target_raw`, and publishes `/servo_node/pose_target_cmds` at a fixed 50 Hz (re-publishing the last target so Servo's 0.1 s `incoming_command_timeout` never trips while holding). Rules, all parameters in `task.yaml`:
   - overwrite z with the fixed push height and orientation with the fixed tool-down quaternion;
   - clamp x, y to the push-area box plus margin;
   - limit the per-cycle step (default 5 mm at 50 Hz = 0.25 m/s) and the distance from the measured EEF pose (default 0.03 m, otherwise hold);
   - watchdog: if the input is stale for more than 0.5 s or the dead-man input is released in policy mode, hold the last pose; on a fault, stop publishing so Servo halts;
   - set Servo to POSE mode through `/servo_node/switch_command_type` (value 2) at startup and re-check every second;
   - publish `/pusht/target_guarded` (what the recorder stores as `action`) and `/pusht/guard_state` (diagnostics).
4. `episode_recorder` samples the latest image, EEF pose, joints, guarded target, block pose and coverage at exactly 10 Hz into a `LeRobotDataset` (features from 3.2), one episode at a time, with `save_episode` on success or operator command and `discard` otherwise.
5. `episode_manager` drives the loop: `reset` (sim: publish a new T pose to `/pusht/t_block/pose_cmd` from the seeded sampler and stream the EEF home through the guard; real: show the start-mark number and wait for the operator), `record`, `stop` (on coverage >= 0.95, on the 300-step limit, or on a key), `score`, `save` or `discard`. Keyboard (`r`, space, `s`, `d`, `q`) and a `std_srvs/Trigger` service for scripted evaluation.
6. Dataset check after each session: `lerobot-dataset-viz` on three random episodes, statistics (episode length, success rate, coverage curve), and a replay test (feed recorded actions through the guard in sim and confirm the T ends near the recorded pose).

Collection targets: 150 to 200 sim episodes (keep successful ones; log the discard rate), 100 to 150 real episodes, 10 to 25 s each. Collect across the full start-state distribution and, on the real robot, under at least two lighting conditions.

### 3.5 Policy execution (runner) and safety layers

`policy_runner` (host venv) loads a checkpoint with `from_pretrained`, builds the observation at 10 Hz (crop and resize the latest `/image_top/rgb`, EEF x,y from `/wmx/moveit2/eef_pose`), calls `predict_action_chunk`, and hands the next `n_action_steps` targets (8 of 16 for Diffusion and Flow Matching, policy-specific for ACT) to the guard as a timed list. The guard interpolates linearly between the 10 Hz waypoints to produce 50 Hz Servo targets, applying the same clamps as in teleop. Inference for the next chunk starts while the current one executes, so an inference time up to about 0.8 s is tolerated without the arm stopping.

Modes: `shadow` (actions computed and logged to `/pusht/policy_target_shadow` while a human teleoperates, no motion), `slow` (Servo `scale.linear` 0.1 and guard step 2 mm), `normal`. Arguments: policy type, checkpoint path, `fm_steps`, `n_action_steps`, `mode`, start-state id.

Safety layers on the real arm, from outside in:

1. Physical E-stop within reach of the operator; the operator holds the Quest grip (or a keyboard key) as a dead-man during policy runs; releasing it makes the guard hold.
2. WMX soft limits enabled in `cr3a_wmx_parameters.xml` for each axis, derived from the joint envelope recorded during sim teleop plus margin.
3. MoveIt Servo joint limits and self-collision checking (on). The table is **not** added as a collision object: Servo's scene proximity threshold is 0.02 m and the tip hovers 10 mm above the table, which would stall every motion. The z lock in the guard and the soft limits cover the table.
4. The guard's workspace clamp, step limit, EEF-distance limit and watchdog.
5. Shadow mode before the first autonomous run of every new checkpoint, then slow mode, then normal.

### 3.6 Python environment

One uv-managed venv on the system Python 3.12 with `--system-site-packages` so that Jazzy's `rclpy` (built for 3.12) is importable next to LeRobot. The venv's own `numpy 2.x` shadows the system 1.26, which is why the venv nodes never import `cv_bridge`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh            # uv is not installed yet
source /opt/ros/jazzy/setup.bash
cd ~/workspaces/movensys-imitation
uv venv --python 3.12 --system-site-packages .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "lerobot[pusht,diffusers-dep,dataset]==0.6.1" shapely rerun-sdk pygame
uv pip install -e .
python -c "import torch, lerobot, rclpy, numpy; print(torch.cuda.is_available(), torch.__version__, lerobot.__version__, numpy.__version__)"
```

Expected: `True 2.x+cu128 0.6.1 2.x` with no import error from `rclpy`. Pin the resolved versions in `uv.lock`. The project `env.bash` sets `ROS_DOMAIN_ID=70`, `MANIPULATOR_MODEL=dobot_cr3a`, sources Jazzy and `movensys_ws/install/setup.bash`, and activates the venv.

GPU sharing: Isaac Sim (about 6 to 10 GB) and policy inference (1 to 2 GB) fit next to each other on the 5090. Training (8 to 16 GB) fits too but slows the simulator, so training runs are scheduled when the simulator is idle (overnight).

### 3.7 Policies and the fairness protocol

| | ACT | Diffusion Policy | Flow Matching (ours, `flow_matching_unet`) | DiT-FM (LeRobot `multi_task_dit`, optional extra) |
|---|---|---|---|---|
| Image encoder | ResNet18 backbone (ImageNet init) + transformer | ResNet18 + spatial softmax (group norm) | identical to Diffusion (imported, not copied) | CLIP ViT-B/16 (pretrained, lr x0.1) |
| Input image | 96 x 96 | 84 x 84 random crop of 96 x 96 (centre crop at eval) | same as Diffusion | 224 x 224 crop of 256 |
| `n_obs_steps` | 1 | 2 | 2 | 2 |
| Horizon / executed | default: chunk 100, temporal ensembling (exec 1 per step, coeff 0.01); matched run: chunk 16 / exec 8 | 16 / 8 | 16 / 8 | 16 / 8 (defaults are 32 / 24) |
| Objective | L1 + VAE KL (weight 10); also a no-VAE run | DDPM epsilon prediction, 100 train steps; DDIM 10 steps at inference (100 as reference) | Conditional OT path `x_t = (1 - t) * noise + t * a`, velocity target `a - noise`, MSE; Euler K in {1, 2, 4, 8, 16}, midpoint optional; `t` uniform or pi0-style Beta(1.5, 1) | Flow matching, beta timestep sampling, Euler 10 steps |
| Normalisation | MIN_MAX state and action, MEAN_STD images (set the same for all) | same | same | same |
| Budget | 100 k steps, batch 64 (200 k for the gym-pusht reference run to match the model card) | same | same | same, 1 seed |
| Seeds | 3 | 3 | 3 | 1 |
| Expected inference (5090) | single pass, a few ms | DDIM 10: tens of ms; DDPM 100: >100 ms | Euler 4 to 8: about 10 ms | tens of ms |

**Flow Matching implementation (the core new code).** `movensys_imitation/policies/flow_matching_unet/` with `configuration_flow_matching_unet.py` (`FlowMatchingUnetConfig`, registered as `flow_matching_unet`, same fields as `DiffusionConfig` minus the scheduler fields, plus `fm_num_steps`, `fm_solver`, `fm_time_sampling`, `fm_sigma_min`), `modeling_flow_matching_unet.py` (`FlowMatchingUnetPolicy`, reusing `DiffusionRgbEncoder` and `DiffusionConditionalUnet1d` from LeRobot; the timestep embedding receives `t * 1000`) and `processor_flow_matching_unet.py` (`make_flow_matching_unet_pre_post_processors`, delegating to the diffusion ones). Entry points `movensys-train` and `movensys-eval` import the package (which triggers registration) and call LeRobot's train and eval mains, so every LeRobot CLI flag keeps working. The type name is deliberately not `flow_matching` to avoid a future clash with an upstream policy of that name.

Unit tests: tensor shapes; loss decreases on a synthetic dataset; one-step and sixteen-step samplers are finite and deterministic under a seed; the velocity field on a 2-D toy problem matches Meta's `flow_matching` library (the team's Flow Matching notes are in `Sync/Research/Diffusion/Flow_Matching`).

**Fairness protocol.** Same dataset and split, same image resolution and augmentation where the method allows, same normalisation, same horizon where the method allows (plus the method's own default), same gradient-step and batch budget, three training seeds, fixed evaluation start states (50 in sim, 20 on the real robot), mean and standard deviation reported, wall-clock training time, peak VRAM and per-chunk inference latency recorded on the same GPU. Hyperparameters are frozen after Phase 1 and not re-tuned on sim or real data.

**Optional extras, decided in Phase 4 if time allows.** (a) DiT-FM baseline (1 seed) to show a transformer flow-matching policy next to ours. (b) LeRobot's real-time chunking (`rtc`, guided mode) on top of our Flow Matching policy, which removes chunk-boundary discontinuities and is directly relevant to the WMX interpolation story in the demo proposals. (c) SmolVLA fine-tune, which the team has run before through Physical AI Studio.

### 3.8 Simulation design (Isaac Sim)

Scene `9a_pusht_simulation.usd` (and `9b_pusht_hil.usd` with the `/isaacsim` namespace), built from `3a_trajectory_simulation.usd`:

- `/World/pusht/t_block`: rigid body made of two boxes (3.1), physics material with friction tuned so that the block slides without tipping; textured top with the tag image.
- `/World/pusht/target_sheet`: thin static plane with the printed sheet texture (T outline, tag id 20, start marks).
- `/World/movensys_manipulator/.../Link6/pusher_tip`: cylinder, collision enabled, no gripper motion needed in sim beyond closing once.
- `/World/camera_top_link`: RealSense model and ROS 2 camera graph copied from `7b_robopoly_hil.usd` (at (0.35, 0, 0.7) looking down), resolution set to 640 x 480, frame skipping for 30 Hz, publishing `/image_top/rgb` and `/image_top/camera_info` with frame `camera_top_color_optical_frame`.
- OmniGraph: `ROS2PublishJointState` `/joint_states`, `ROS2SubscribeJointState` `/joint_command` (unchanged), `/clock`, plus a pose publisher for the T (`/pusht/t_block/pose`, `geometry_msgs/Pose`, 30 Hz) and a pose-command subscriber (`/pusht/t_block/pose_cmd`) that teleports the T and zeroes its velocity, copied from the Monopoly piece graphs.
- The hand camera is left out for now (ablation option later).

Reset is therefore a pair of messages from the host (`pose_cmd` for the T and a planar move for the EEF through the guard), which works in the GUI workflow the team already uses. Scene files are contributed back to `movensys-simulation` as `9a`/`9b` once stable; the T and sheet assets live in this repo under `sim/isaacsim/assets/`.

Scoring in sim uses the ground-truth T pose from the publisher, expressed relative to the sheet prim.

### 3.9 Real-robot design

- **Hardware to make or buy (small budget):** printed T block (PLA, solid), printed pusher tip, laminated target sheet with tags and start marks, a camera mount above the push area for the existing "nvblox" RealSense (or one additional D435 if the nvblox camera must stay where it is). No force sensor is needed for PushT.
- **Cameras:** overhead RealSense at 640 x 480, 30 fps, colour only, launched as `/image_top/{rgb,camera_info}` (new launch file in this repo modelled on `camera_nvblox.launch.py`); the hand camera profile is raised from 6 to 30 fps if the wrist-view ablation is run.
- **Block pose:** a second `apriltag_detector` instance configured for `/image_top/rgb` with tag sizes 0.04 m (id 10, T) and 0.05 m (id 20, sheet). The scorer reads TF `camera_top_color_optical_frame -> tag36h11:10` and `-> tag36h11:20` and computes the T pose in the sheet frame, so camera-to-robot calibration is not needed for scoring. The image crop is anchored on the sheet tag's pixel corners so the framing matches the sim scene.
- **Workspace alignment (once):** jog the tip to three marks on the sheet, read `/wmx/moveit2/get_eef_pose`, and store the sheet-to-base transform in `config/dobot_cr3a/real_calibration.yaml`. It is used for the start-state sampler (which mark corresponds to which base-frame pose) and for sim-real comparison, not by the policy.
- **WMX-R2:** `wmx_r2_cr3a_manipulator.launch.py` plus the streaming controller chosen in Phase 0. Soft limits enabled. Never call `/wmx/axis/homing` on the CR3A (it redefines zero on the absolute encoders). Always send at least two points to the `FollowJointTrajectory` action if it is used at all (a one-point goal reads out of bounds on `main`).
- **Reset:** the operator places the T on the printed start mark shown by the episode manager; the EEF homes through the guard.

---

## 4. Repository layout

```
movensys-imitation/
├── README.md  LICENSE.txt (MIT)  .gitignore  pyproject.toml  uv.lock  env.bash
├── .github/workflows/lint.yml           # ruff + ament_flake8, same style as the sibling repos
├── doc/
│   ├── plan.md                          # this document
│   ├── 1_setup.md                       # venv, containers, env vars, domain id, CloudXR
│   ├── 2_task_pusht.md                  # geometry, scoring, printable assets, start marks
│   ├── 3_gym_pusht.md                   # stage 1 commands and results
│   ├── 4_sim.md                         # scene, teleop in sim, collection, training, evaluation
│   ├── 5_hil.md                         # WMX simu platform + Isaac 9b scene
│   ├── 6_real.md                        # hardware, calibration, safety checklist, collection, evaluation
│   └── 7_results.md                     # comparison tables, videos, failure taxonomy
├── movensys_imitation/                  # pure Python library, no ROS imports
│   ├── policies/flow_matching_unet/     # configuration_, modeling_, processor_ (Section 3.7)
│   ├── tasks/pusht/                     # geometry.py (T, target polygons), scoring.py (coverage), sampler.py (start states), crop.py (tag-anchored crop)
│   ├── data/                            # LeRobotDataset writer helpers, gym-pusht conversion, stats, replay check
│   ├── configs/                         # act.yaml diffusion.yaml flow_matching.yaml dit_fm.yaml pusht_gym.yaml pusht_sim.yaml pusht_real.yaml
│   ├── cli/                             # movensys-train, movensys-eval (register policies, then call LeRobot)
│   ├── compare.py                       # builds the results tables and plots from eval CSVs
│   └── tests/
├── movensys_imitation_ros/              # ament_python ROS 2 package, run from the host venv
│   ├── movensys_imitation_ros/
│   │   ├── planar_guard.py              # Section 3.4 step 3
│   │   ├── episode_recorder.py          # topics -> LeRobotDataset at 10 Hz
│   │   ├── episode_manager.py           # reset / record / score / save loop, keyboard + service
│   │   ├── pusht_scorer.py              # coverage from the sim pose topic or the AprilTag TFs
│   │   ├── policy_runner.py             # checkpoint -> 10 Hz chunks -> guard; shadow / slow / normal
│   │   ├── eval_sim.py  eval_real.py    # fixed start-state sweeps, CSV + video per episode
│   │   └── image_utils.py               # numpy Image decode, crop, resize (no cv_bridge)
│   ├── launch/collect.launch.py  run_policy.launch.py  cameras_real.launch.py   (arg mode:=sim|hil|real)
│   └── config/dobot_cr3a/task.yaml  guard.yaml  recorder.yaml  real_calibration.yaml  apriltag_top.yaml
├── sim/isaacsim/
│   ├── 9a_pusht_simulation.usd  9b_pusht_hil.usd      # until contributed to movensys-simulation
│   └── assets/t_block.usd  target_sheet.png  pusher_tip.usd
├── hardware/                            # t_block.stl  pusher_tip.stl  target_sheet.pdf (tags + start marks)  camera_mount.md
└── scripts/                             # setup_venv.sh  smoke_test_gym.sh  run_sweep.sh  make_tables.sh
```

Data and checkpoints live outside git: `datasets/` and `outputs/` are ignored; dataset ids are `movensys/pusht_gym` (converted public data), `movensys/pusht_cr3a_sim_v1`, `movensys/pusht_cr3a_real_v1`, optionally mirrored to a private Hugging Face organisation.

---

## 5. Work plan

Task owners: **C** = controls / robotics engineer, **M** = ML engineer. With one person, do them in the listed order; the C and M tasks inside a phase are independent.

### Phase 0: environment and prerequisites in the sibling repos (week 1)

| # | Task | Owner | Notes |
|---|---|---|---|
| 0.1 | Create the GitHub repo `movensys/movensys-imitation` with the layout in Section 4, MIT licence, lint workflow, `.gitignore` for `datasets/`, `outputs/`, `.venv/`, `wandb/`. | M | |
| 0.2 | Build the venv (Section 3.6) and run the LeRobot smoke test: 500 training steps of Diffusion on `lerobot/pusht`, then `lerobot-eval` with 5 episodes on gym-pusht. | M | Proves cu128 torch on the 5090, LeRobot 0.6.1, gym-pusht and `rclpy` co-exist. |
| 0.3 | Switch `movensys-manipulator` to `origin/devel` (`movensys-teleoperation` is already on `devel` @ 50f91f7); make sure the shell that runs `docker compose` has `ROS_DOMAIN_ID=70` (both compose files pass `${ROS_DOMAIN_ID}` from the host environment, so the `~/.bashrc` value is what counts; ignore the 73 in the manipulator README); add `ros-jazzy-moveit-servo` to the `isaac-ros_4.1` Dockerfile; build and start `movensys_manipulator_container`; confirm `moveit.launch.py` brings up `servo_node` and that `ros2 service type /servo_node/switch_command_type` returns `moveit_msgs/srv/ServoCommandType`. | C | |
| 0.4 | Open `3a_trajectory_simulation.usd` in Isaac Sim 5.1 with Jazzy sourced, start `sim_bridge.launch.py use_sim_time:=true`, `moveit.launch.py use_sim_time:=true`, and move the arm with `keyboard_teleop` in `p` mode. | C | Verifies the existing sim path on this host (never done here: no containers exist yet). |
| 0.5 | Teleop container: build, download the CloudXR runtime from NGC, run `quest_pose.launch.py`, connect the Quest 2 through the WebXR client, and confirm `/quest_pose_publisher/controller_pose_right` at about 55 Hz. Then run `quest_servo_teleop.launch.py` in sim with `motion_mode:=planar`. | C | Known issues are listed in the teleop repo's `doc/1_setup.md` to `doc/3_quest_pose.md` (UID/GID env, compose >= 2.30 for `gpus: all`, seeded client URL, Quest drops Wi-Fi when not worn). |
| 0.6 | **wmx-r2 streaming controller decision.** Compare `joint_position_controller` and `servo_stream_controller` (`origin/feature/api-buffer-command-streaming*`) and the cyclic-buffer branch; pick one for the first real runs (recommendation: `joint_position_controller`, which the VR teleop already used), rebase it on `main`, open a PR. Also fix the gripper node's hard-coded `/opt/lmx/` path. | C | Blocks Phase 2b and 3. The branch doc says the buffered six-axis path is unverified on hardware. |
| 0.7 | WMX simu platform for HIL: extend `/opt/wmx3/platform/simu/simu_network.def` to six axes, launch `wmx_r2_cr3a_manipulator.launch.py` with the streaming controller, publish a hand-made `JointTrajectory` on `/movensys_manipulator_arm_controller/joint_trajectory` and watch `/joint_states` follow. Note the free licence's 1 h communication limit per session. | C | |
| 0.8 | Write `doc/1_setup.md` with everything above, including the exact `sudo --preserve-env` launch line for wmx-r2. | C | |

Exit: train and eval commands run on gym-pusht from the venv; Servo moves the CR3A in Isaac Sim from the keyboard and from the Quest; a `JointTrajectory` moves the WMX simu axes; the streaming-controller PR exists.

### Phase 1: Flow Matching policy and the gym-pusht benchmark (weeks 2 to 3)

| # | Task | Owner |
|---|---|---|
| 1.1 | Implement `flow_matching_unet` (Section 3.7) with unit tests and the 2-D toy cross-check. | M |
| 1.2 | Write `movensys-train` / `movensys-eval` entry points and the three policy YAMLs; confirm `--policy.type=flow_matching_unet` resolves through LeRobot's factory. | M |
| 1.3 | Reference run: Diffusion on `lerobot/pusht`, 200 k steps, 50 eval episodes, compare with the model card (65 % band). | M |
| 1.4 | Benchmark: ACT (default and matched horizon, with and without VAE), Diffusion (DDIM 10 and DDPM 100), Flow Matching (Euler 1, 2, 4, 8, 16), 100 k steps, batch 64, 3 seeds, 50 episodes each, fixed eval seeds. Record training time, VRAM and latency. | M |
| 1.5 | `compare.py` produces the tables and plots; write `doc/3_gym_pusht.md`; freeze hyperparameters for the robot phases. | M |
| 1.6 | Convert `lerobot/pusht` into our schema (Section 3.2, pixel units) with `data/convert_gym_pusht.py` to prove the recorder's feature definitions train. | M |

Exit: the three policies train and evaluate in one harness; Diffusion reproduces the published range; Flow Matching with 4 to 8 Euler steps is within noise of Diffusion; the frozen configs are committed.

### Phase 2: PushT in Isaac Sim with the CR3A (weeks 3 to 6)

| # | Task | Owner |
|---|---|---|
| 2.1 | Build `9a_pusht_simulation.usd` (Section 3.8): T block, sheet, pusher tip, overhead camera at 30 Hz, T pose publisher and pose-command subscriber. Fix the push-area centre after a reach check at the push height. | C |
| 2.2 | `planar_guard` with tests (clamp, step limit, watchdog, command-type switch); `task.yaml` and `guard.yaml`. Teleop launch with the `quest_servo_teleop` remap. | C |
| 2.3 | `pusht_scorer` (sim mode) and `tasks/pusht/scoring.py` with shapely; unit tests against hand-computed coverages. | M |
| 2.4 | `episode_recorder` and `episode_manager`; `collect.launch.py mode:=sim`. Pilot: 10 episodes, visualise with `lerobot-dataset-viz`, run the replay check. | M + C |
| 2.5 | Operator tuning session: `position_scale`, guard step, camera crop, tip height; write the operator guide in `doc/4_sim.md`. | C |
| 2.6 | Collect 150 to 200 episodes across seeds 1000+; keep successes; record the discard rate. | C |
| 2.7 | Train the three policies (3 seeds) with the Phase 1 configs; about 1 to 2 h each on the 5090 while the simulator is idle. | M |
| 2.8 | `policy_runner` with shadow, slow and normal modes; `run_policy.launch.py mode:=sim`. First autonomous episode in shadow mode, then normal. | M |
| 2.9 | `eval_sim.py`: 50 fixed start states x 3 policies x 3 seeds (about 2.5 h of real-time simulation per sweep), CSV + video per episode; ablation runs for Flow Matching step count. Fill `doc/7_results.md` (sim section). | M |

Exit: measured success rates for the three policies in Isaac Sim on the fixed start-state set; collect and run launch files work with `mode:=sim`.

### Phase 2b: hardware-in-the-loop through WMX (week 6, overlaps with 2.9)

| # | Task | Owner |
|---|---|---|
| 2b.1 | `9b_pusht_hil.usd` (`/isaacsim` namespace). Launch: Isaac Sim `9b`, wmx-r2 on the simu platform with the streaming controller and the `/joint_states -> /isaacsim/joint_command` relay, manipulator container with `sim_bridge` disabled, guard, runner. | C |
| 2b.2 | Run 10 teleop episodes and 10 policy episodes through this path; compare commanded vs WMX-reported joints (tracking error, step timing) and the Servo-to-WMX latency. Fix controller timing issues here, not on hardware. | C |

Exit: a policy episode succeeds in Isaac Sim with WMX-R2 in the loop; the streaming controller PR is merged.

### Phase 3: real CR3A (weeks 7 to 9)

| # | Task | Owner |
|---|---|---|
| 3.1 | Hardware: print T and tip, laminate the sheet, mount the overhead camera, confirm EtherCAT is enabled in `/opt/wmx3/Module.ini` on the robot PC, license valid, E-stop tested. | C |
| 3.2 | Enable soft limits in `cr3a_wmx_parameters.xml` from the Phase 2 joint envelope plus margin; verify Servo joint limits; verify the gripper service closes on the tip. | C |
| 3.3 | Cameras and tags: `cameras_real.launch.py`, AprilTag on `/image_top/rgb`, scorer in real mode; workspace alignment with three marks (Section 3.9). | C |
| 3.4 | Low-speed path check: Servo `scale.linear` 0.1, guard step 2 mm, Quest teleop in `mode:=real` (`use_sim_time:=false` everywhere); verify the z lock, the clamp, the watchdog and the E-stop; compare `/wmx/moveit2/eef_pose` with the sim frames so recorder and runner run unchanged. | C |
| 3.5 | Collect 100 to 150 real episodes using the numbered start marks, two lighting setups. | C |
| 3.6 | Train the three policies on real data (3 seeds); optional: sim + real co-training and fine-tuning from the sim checkpoint. | M |
| 3.7 | Deploy: shadow mode per checkpoint, then slow, then normal; 20 trials per policy (best seed by sim score, chosen before real eval) on the same 20 start marks; record with the overhead camera and `rosbag2video`; collect WMX-side tracking error and timing per trial. | M + C |
| 3.8 | `doc/6_real.md` with the safety checklist and the procedure; `doc/7_results.md` real section. | M + C |

Exit: three policies run on the real CR3A through WMX-R2 with measured success rates; the procedure is documented.

### Phase 4: analysis and write-up (week 10)

Ablations as time allows, in priority order: Flow Matching step count on real data; action horizon (16/8 vs 32/16); observation history (1 vs 2); image resolution (96 vs 128); the DiT-FM extra baseline; `rtc` on the Flow Matching policy; wrist camera on/off. Failure taxonomy per policy (stuck at the edge, overshoot, wrong side of the T, oscillation at chunk boundaries). Final tables, plots and a short video montage in `doc/7_results.md`; a one-page summary for management reusing the format of the existing demo proposals.

---

## 6. Evaluation protocol and metrics

| Metric | Definition | Where |
|---|---|---|
| Success rate | coverage >= 0.95 at any step within 300 steps (10 Hz) | all stages |
| Success at 0.90 | secondary threshold, same episodes | all stages |
| Average max coverage | max coverage per episode, averaged | all stages |
| Time to success | steps until the threshold, successful episodes only | all stages |
| Path length | integrated pusher travel (m) | all stages |
| Smoothness | mean absolute jerk of the commanded targets; on hardware also RMS joint tracking error from `/joint_states` vs commanded | sim, HIL, real |
| Inference latency | wall-clock per chunk on the 5090 and on CPU, mean and p95 | all stages |
| Training cost | gradient steps, wall-clock, peak VRAM | all stages |
| Operator cost | episodes collected per hour, discard rate | sim, real |

Protocol: fixed start-state lists (gym seeds, sim sampler seeds 0 to 49, 20 numbered real marks); the same list for every policy and seed; mean and standard deviation over training seeds; videos kept for every real trial and for failures in sim; results committed as CSV so tables are regenerated by `compare.py`.

---

## 7. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The wmx-r2 streaming controller is not merged or misbehaves on six axes | Medium | High: blocks real-robot execution | Decide in Phase 0 (task 0.6); HIL stage (2b) exercises it on the simu platform before hardware; `joint_position_controller` first, buffered path as an experiment |
| Quest/CloudXR bring-up on this host takes longer than planned (runtime download, Wi-Fi, headset drop-outs) | Medium | Medium: delays collection | Start in week 1 (task 0.5); fallback demonstration sources that need no new hardware: `keyboard_teleop` `p` mode for pilots, a mouse-on-overhead-image teleop node (small, same modality as the original PushT data) if the Quest is unavailable |
| Human demonstrations through Servo are too slow or jerky, producing a poor dataset | Medium | High | Pilot sessions and operator tuning (2.5); replay check; keep only successful episodes; `position_scale` 1:1; raise Servo linear scale after the first sessions |
| ACT scores near zero on PushT (multimodality) | High | Low for the project, since it is a finding | Run ACT with matched horizon, with and without VAE and with temporal ensembling; report all variants |
| Sim-to-real gap in images (lighting, texture) | High | Medium | Policies for the real robot train on real data; the sim stage validates the pipeline; identical crop anchoring on the sheet tag; two lighting conditions in the real dataset |
| Servo collision checking or thresholds stall motion near the table | Medium | Medium | No table collision object; z lock in the guard; soft limits in WMX; tip height 10 mm |
| GPU contention between Isaac Sim, inference and training | Medium | Low | Train when the simulator is idle; inference next to the simulator is fine on 32 GB |
| `cv_bridge` or other system Python packages break under numpy 2 in the venv | High | Low | Never import them in venv nodes; numpy decoding of images; only `rclpy`, `tf2_ros` and message packages from the system |
| Isaac Sim 5.1 is a release candidate on this host | Low | Medium | Scenes are plain USD + OmniGraph; moving to 5.1.0 final or 6.x is a re-open and re-save |
| `ROS_DOMAIN_ID` mismatch (70 vs 73) or `use_sim_time` mismatch between host, containers and Isaac | High | Low once known | Containers inherit `ROS_DOMAIN_ID` from the host shell (70); this repo's `env.bash` and launch files set the same value and the right `use_sim_time`; `doc/1_setup.md` has a checklist |
| Real-robot damage from a bad policy action | Low | High | Section 3.5 safety layers; shadow and slow modes; E-stop; soft limits; step and distance limits in the guard |
| Disk use from checkpoints and videos | Low | Low | 201 GB free; keep top-2 checkpoints per run; datasets are video-encoded |

---

## 8. Timeline and milestones

Single engineer, starting 2026-10-13. With a controls engineer (C tasks) and an ML engineer (M tasks) in parallel, Phase 1 overlaps Phases 0 and 2, and M2 moves to week 5 and M3 to week 8.

| Week | Dates | Phase | Milestone |
|---|---|---|---|
| 1 | Oct 13 to 17 | 0 | **M0:** venv + gym-pusht smoke test; Servo moves the CR3A in Isaac Sim on this host; streaming-controller decision |
| 2 | Oct 20 to 24 | 1 | Flow Matching policy implemented and unit-tested |
| 3 | Oct 27 to 31 | 1, start 2 | **M1:** gym-pusht comparison table (3 policies x 3 seeds), configs frozen; Isaac PushT scene and guard started |
| 4 | Nov 3 to 7 | 2 | Quest teleop in the PushT scene, recorder, episode manager, pilot episodes |
| 5 | Nov 10 to 14 | 2 | Sim dataset (150+), training |
| 6 | Nov 17 to 21 | 2, 2b | **M2:** sim evaluation table; HIL episode through WMX-R2; controller PR merged |
| 7 | Nov 24 to 28 | 3 | Hardware prepared, soft limits, cameras and tags, low-speed teleop check |
| 8 | Dec 1 to 5 | 3 | Real dataset, training |
| 9 | Dec 8 to 12 | 3 | **M3:** real evaluation (20 trials x 3 policies) |
| 10 | Dec 15 to 19 | 4 | **M4:** ablations, failure taxonomy, results document, summary slide |

---

## 9. Decisions needed and assumptions

Decisions requested from the team before week 2:

1. **Robot and PC.** Confirm the Dobot CR3A and that this workstation is also the robot PC (WMX3 licence and EtherCAT NIC here). If the robot PC is another 24.04 machine, Phase 3 moves wmx-r2 and the runner there and the policy trains here.
2. **wmx-r2 branch.** Who owns the streaming-controller merge (task 0.6) and which controller is the first choice.
3. **Overhead camera.** Re-mount the nvblox RealSense above the push area, or buy one more D435 (about KRW 0.5 M) and keep the current setup intact.
4. **Scene ownership.** Contribute `9a`/`9b` to `movensys-simulation` (recommended) or keep them in this repo.
5. **Extras.** Include the DiT-FM baseline (recommended, 1 seed, about half a day of GPU) and the `rtc` ablation; SmolVLA fine-tuning only if a stakeholder asks for it.
6. **Dataset hosting.** Local only, or a private Hugging Face organisation for sharing with Intel or other partners.

Assumptions this plan makes (correct me if wrong):

- The Quest 2 used for the VR teleop work is available, and the team member who did that bring-up can support task 0.5.
- The T block can be pushed at 10 mm tip height without the fingers touching the table (fingers are 20 mm tall when closed on the tip); otherwise the tip gets longer.
- gym-pusht's success threshold (0.95) is acceptable for the real task; if the printed T and tags make 0.95 too strict, we report 0.90 as primary and say so.
- No force/torque sensor and no gripper width control are needed; the gripper only holds the tip.
- One operator collects all demonstrations in a phase, so operator style does not become a hidden variable between sim and real datasets.

---

## Appendix A. Interface cheat sheet

| Interface | Type | Direction / owner | Notes |
|---|---|---|---|
| `/quest_pose_publisher/controller_pose_right` | `geometry_msgs/PoseStamped` (frame `quest_world`) | teleop container | about 55 Hz; not published while the controller is untracked |
| `/quest_pose_publisher/joy_right` | `sensor_msgs/Joy` | teleop container | axes `[thumb_x, thumb_y, trigger, squeeze]`, buttons `[primary, secondary, thumbstick, menu]`; squeeze >= 0.6 is the clutch |
| `/pusht/target_raw` | `PoseStamped` | `quest_servo_teleop` (remapped) or `policy_runner` | 50 Hz while clutched |
| `/servo_node/pose_target_cmds` | `PoseStamped` (frame `world_manipulator`) | `planar_guard` | 50 Hz, always |
| `/servo_node/switch_command_type` | `moveit_msgs/srv/ServoCommandType` | guard calls it | 0 JOINT_JOG, 1 TWIST, 2 POSE |
| `/movensys_manipulator_arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | `servo_node` (40 Hz) | consumed by `sim_bridge` (sim) or the wmx-r2 streaming controller (HIL, real) |
| `/joint_command`, `/joint_states` | `sensor_msgs/JointState` | Isaac Sim scene (sim); `/isaacsim/...` in HIL | |
| `/joint_states` | `sensor_msgs/JointState` | wmx-r2 (100 Hz) | includes `picker_1_joint`, `picker_2_joint` |
| `/wmx/moveit2/eef_pose` | `PoseStamped` (frame `world_manipulator`) | `moveit2_api` (100 Hz, devel) | `observation.state` source |
| `/wmx/set_gripper` | `std_srvs/SetBool` | `sim_bridge` (sim) / `gripper_controller` (real) | true = close on the tip |
| `/image_top/rgb`, `/image_top/camera_info` | `sensor_msgs/Image`, `CameraInfo` | Isaac camera graph (sim) / RealSense (real) | 640 x 480, 30 Hz, frame `camera_top_color_optical_frame` |
| `/pusht/t_block/pose` | `geometry_msgs/Pose` | Isaac pose publisher (sim) | 30 Hz, ground truth |
| `/pusht/t_block/pose_cmd` | `geometry_msgs/Pose` | `episode_manager` -> Isaac subscriber graph | teleports the T on reset |
| TF `camera_top_color_optical_frame -> tag36h11:10` / `:20` | TF | `apriltag_detector` on the overhead image (real) | T and sheet tags |
| `/pusht/target_guarded`, `/pusht/guard_state`, `/pusht/coverage`, `/pusht/policy_target_shadow` | this repo | guard, scorer, runner | recorded / diagnostics |
| `/wmx/engine/ready`, `/wmx/core_motion/ready` | `std_msgs/Bool` (transient local) | wmx-r2 | wait for both before streaming |
| `/wmx/axis/clear_alarm`, `/wmx/axis/set_on`, `/wmx/axis/stop` | `wmx_r2_message/srv/SetAxis` | wmx-r2 | recovery; never `homing` on the CR3A |

Frames: `world_manipulator` (planning frame; `Link0` at (-0.03, 0.275, 0.05), yaw -90 deg), `Link6` (Servo `ee_frame`, no TCP link exists), table top at z about 0.043 in sim. Joints: `joint1..joint6`, gripper `picker_1_joint`, `picker_2_joint` (0 to 0.045 m).

## Appendix B. Command reference (to be completed in `doc/1_setup.md`)

```bash
# gym-pusht smoke test (Phase 0)
movensys-train --policy.type=diffusion --env.type=pusht --dataset.repo_id=lerobot/pusht \
  --steps=500 --batch_size=64 --output_dir=outputs/smoke_diffusion
movensys-eval --policy.path=outputs/smoke_diffusion/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=5 --eval.batch_size=5

# our policy (Phase 1)
movensys-train --policy.type=flow_matching_unet --policy.fm_num_steps=8 \
  --env.type=pusht --dataset.repo_id=lerobot/pusht --steps=100000 --batch_size=64 --seed=1

# simulation session (Phase 2), one terminal each
#  1. Isaac Sim: ~/isaac-sim/isaac-sim.sh, open sim/isaacsim/9a_pusht_simulation.usd, Play
#  2. mros: ros2 launch movensys_manipulator_moveit_config sim_bridge.launch.py use_sim_time:=true
#  3. mros: ros2 launch movensys_manipulator_moveit_config moveit.launch.py use_sim_time:=true
#  4. mtele: ros2 launch quest_pose_publisher quest_pose.launch.py
#  5. mros: ros2 launch movensys_manipulator_moveit_config quest_servo_teleop.launch.py motion_mode:=planar \
#           --ros-args -r /servo_node/pose_target_cmds:=/pusht/target_raw
#  6. host venv: ros2 launch movensys_imitation_ros collect.launch.py mode:=sim
#  policy run: ros2 launch movensys_imitation_ros run_policy.launch.py mode:=sim policy:=flow_matching_unet \
#              checkpoint:=outputs/sim_fm_s1/checkpoints/last/pretrained_model run_mode:=shadow

# real robot (Phase 3): wmx-r2 first, as root, then the container stack with use_sim_time:=false
sudo --preserve-env=PATH,AMENT_PREFIX_PATH,COLCON_PREFIX_PATH,PYTHONPATH,LD_LIBRARY_PATH,ROS_DISTRO,ROS_VERSION,ROS_PYTHON_VERSION,ROS_DOMAIN_ID,RMW_IMPLEMENTATION \
  bash -c "source /opt/ros/jazzy/setup.bash && source ~/workspaces/movensys_ws/install/setup.bash && \
           ros2 launch wmx_r2_package wmx_r2_cr3a_manipulator.launch.py"
```

## Appendix C. References

- LeRobot 0.6.1 on PyPI (requirements and extras); policy factory and `PreTrainedPolicy` in `src/lerobot/policies/`; `multi_task_dit` and `rtc` configs in the same tree.
- `lerobot/diffusion_pusht` and `lerobot/diffusion_pusht_keypoints` model cards (PushT reference numbers and the 0.95 success definition).
- gym-pusht (`gym_pusht/PushT-v0`): 96 x 96 pixels, 2-D agent position, 2-D absolute target action, 10 Hz, 300-step episodes.
- Chi et al., Diffusion Policy (2023); Zhao et al., ACT / ALOHA (2023); Lipman et al., Flow Matching for Generative Modeling (2023); Black et al., pi0 (2024, flow matching for robot actions); Black et al., Real-Time Chunking (2025). Copies of the first papers are in `Sync/Research/VLA_based_motion_roadmap` and `Sync/Research/Diffusion/Flow_Matching`.
- Isaac Sim 5.1 ROS 2 installation notes (Ubuntu 24.04 + Jazzy pairing, bundled bridge libraries, Python 3.11 interpreter).
- Internal: `Proposal_new_Demo/Physical AI Demo Proposal ... with WMX3` (the 10 Hz policy vs 1 kHz WMX3 narrative that Phase 2b and the smoothness metrics feed), `national_project_vision_based_teaching/VR_based_motion_Real/Meta Quest VR teleop with Dobot CR3A.md`, `intel_collaboration/physical AI Studio/report` (previous ACT and SmolVLA runs).
