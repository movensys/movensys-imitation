# 4. Stage 2: PushT in Isaac Sim

Plan Phase 2. The scene (P2.1) is documented in `sim/isaacsim/README.md`; this file covers the
nodes that run against it, starting with the planar guard (P2.2). Collection, training and
evaluation sections follow with P2.4 to P2.9.

## Planar guard (P2.2)

`planar_guard` (package `movensys_imitation_ros`, host venv) is the only writer to
`/servo_node/pose_target_cmds`. It locks the tool to a horizontal plane and drives the tip in
that plane from the Quest right thumbstick, or from absolute targets when a policy or a replay
is the source. The plane is **latched from the measured `Link6` pose** when the guard starts:
park the rod at push height first, then start the guard. Everything after that is x, y only.

Per 50 Hz cycle (`guard_core.GuardCore.update`): fault if the end-effector pose is missing or
stale (nothing is published, Servo halts on its own timeout); anchor on the measured pose at
start (fault if the tip starts outside the push area by more than the lead limit); re-anchor if
Servo reports a halt; hold on a released dead-man, stale Joy or a centred stick; otherwise
integrate the stick velocity (radial deadzone, quadratic expo, `stick_yaw_deg`, `max_speed`),
clamp to the push area, cap the step at `step_limit`, and hold if the target would lead the
measured end effector by more than `eef_distance_limit`. Because the integrator lives inside the
guard, pushing the stick into an edge does not wind up: reversing the stick moves back at once.

Parameters: `config/dobot_cr3a/task.yaml` (area, frames, table and tip numbers, shared by all
nodes) and `config/dobot_cr3a/guard.yaml` (everything else; comments inline).

### Bring-up in sim

1. Isaac Sim with `9a_pusht_simulation.usd` playing; manipulator container with `sim_bridge`
   (`simulator:=isaacsim`) and `moveit.launch.py` (`use_sim_time:=true`); teleop container with
   `quest_pose_node` (`hand:=both` or `right`), Quest connected. Do **not** start
   `quest_servo_teleop` or `keyboard_teleop` in `p` mode: both write to Servo's pose topic.
2. Put the rod at push height with keyboard teleop or a move_group plan, tip about 10 mm above
   the sheet, tool vertical, inside the push area. Stop that teleop.
3. Build the package once (`env.bash` then sources `install/`), and launch:
   ```bash
   source env.bash
   colcon build --symlink-install --base-paths movensys_imitation_ros --packages-select movensys_imitation_ros
   source env.bash
   ros2 launch movensys_imitation_ros guard.launch.py mode:=sim
   ```
   The log must show `Servo switched to POSE mode` and
   `plane latched: z=... tip clearance ... mm` (a warning if more than 5 mm off nominal; fix the
   height and call `ros2 service call /planar_guard/relatch std_srvs/srv/Trigger`).
4. Hold the right grip (dead-man) and push the stick. Calibrate once: with the stick forward,
   the tip should move away from you; set `stick_yaw_deg` or `stick_invert_*` in `guard.yaml`
   until it does. Then raise `max_speed` to taste (the step limit caps it at 0.25 m/s).
5. Watch `ros2 topic echo /planar_guard/state`: `moving:`, `hold:deadman`, `hold:stick_centred`,
   `hold:eef_distance` (the arm is lagging, usually in Servo's slow-down zone near the near edge
   and far corners), `hold:servo_halted` (singularity or collision stop; back out),
   `fault:eef_stale` (no TF; nothing is published).

Note: colcon writes the `planar_guard` script with `/usr/bin/python3`, which is fine here (the guard
needs only system ROS packages). Nodes that need the venv (recorder, runner, scorer: numpy 2,
torch) must be started as `python -m movensys_imitation_ros.<node>` from the venv instead.

Switch source at runtime: `ros2 param set /planar_guard source pose`; then publish
`geometry_msgs/PoseStamped` on `/pusht/target_raw` (frame `world_manipulator`) or a timed
`nav_msgs/Path` on `/pusht/waypoints` (policy runner, replay). The dead-man still applies unless
`require_deadman` is false (unattended sim evaluation only).

### Dry run without touching Servo

To check latching, the stick mapping and the 50 Hz stream while something else owns the arm,
remap the two Servo interfaces. Remaps always apply; verified 2026-10-24 (target equals the
measured pose, 151 messages in 3 s, Servo kept its single writer):

```bash
ros2 run movensys_imitation_ros planar_guard --ros-args \
  -r /servo_node/pose_target_cmds:=/pusht/guard_dryrun \
  -r /servo_node/switch_command_type:=/pusht/guard_dryrun_switch \
  -p use_sim_time:=true \
  --params-file install/movensys_imitation_ros/share/movensys_imitation_ros/config/dobot_cr3a/task.yaml \
  --params-file install/movensys_imitation_ros/share/movensys_imitation_ros/config/dobot_cr3a/guard.yaml
ros2 topic hz /pusht/guard_dryrun        # 50 Hz
ros2 topic echo /planar_guard/state      # on change only
```

Use remaps (`-r`) or an extra params file for such overrides, not `-p`: with the node named
`planar_guard`, a `-p servo_pose_topic:=...` after `guard.yaml` was ignored (2026-10-24; rcl
applies the exact-name yaml section after the command-line overrides) and the guard streamed to
the real Servo topic. A params file passed last, as the launch file does for `source:=`, wins.

### Troubleshooting

* `fault:eef_stale` at start and nothing else: the guard cannot see TF. Check the ROS domain
  first: every host shell, the shell that ran `docker compose up` and Isaac Sim must agree
  (project domain 0 since 2026-10-24; before that `env.bash` forced 70 while the containers and
  Isaac ran on 0, and the guard sat alone). Compare `echo $ROS_DOMAIN_ID` with
  `docker exec <container> env | grep ROS_DOMAIN`. `ros2 run tf2_ros tf2_echo world_manipulator
  Link6` must print a pose before the guard can latch.
* The guard is unaffected by Isaac's RMW (fastrtps) versus the host and containers (cyclonedds):
  discovery across the two vendors works on the same domain.
* Jittery or fighting motion: a second writer on `/servo_node/pose_target_cmds`. `ros2 topic info
  -v /servo_node/pose_target_cmds` must list only `planar_guard`; stop `quest_servo_teleop` or
  `keyboard_teleop` otherwise.

### Evidence for the done cell

A two-minute stick teleop with: `/pusht/target_guarded` at 50 Hz (`ros2 topic hz`); z of the
published target within 1 mm of the latched value; `hold:` at all four edges; `hold:joy_stale`
after stopping `quest_pose_node`; no `servo_halted` inside the area (this also closes the P2.1
corner check). Unit tests: `pytest movensys_imitation_ros/test/test_guard_core.py`.

## SpaceMouse input (P2.2b)

A 3Dconnexion SpaceMouse (Pro Wireless, used cabled) is a second input next to the Quest,
decided 2026-10-29 on branch `feature/support-for-3d-spacemouse`. It runs on the host through
`spacenav_node` (`ros-jazzy-spacenav`, which talks to the `spacenavd` daemon; install in
`doc/1_setup.md` 1.7) and serves two jobs:

* **Collection**: the puck is the guard's stick instead of the Quest thumbstick. The guard
  output, the dataset and the manager are unchanged; the sidecar records `input_device`.
* **Free motion**: `spacemouse_teleop` jogs the arm in 6-DoF through Servo's TWIST mode, for
  parking the rod at push height and the reach checks, in place of the joint-movement service
  calls or `keyboard_teleop`.

One Servo writer at a time still holds: the guard, `quest_servo_teleop`, `keyboard_teleop` in
`p` or `t` mode and `spacemouse_teleop` never run together. The teleop node checks for a
publisher on `/servo_node/pose_target_cmds` before it switches Servo to TWIST and holds with
`hold:pose_writer` while one exists; the guard switches Servo back to POSE when it starts, so
the order is: park with `spacemouse_teleop`, Ctrl-C it, launch the guard or the collect stack.

### Driver bring-up (once)

```bash
ros2 launch movensys_imitation_ros spacenav.launch.py      # spacenav_node, zero_when_static
ros2 topic hz /spacenav/joy                                 # puck at rest: must stay above 5 Hz
ros2 topic echo /spacenav/joy                               # axes [x, y, z, roll, pitch, yaw], buttons
```

`/spacenav/joy` axes are raw counts over `full_scale`, so about -1..1, and the puck returns to
exactly 0. At rest the node republishes zeros about every 30 ms (`zero_when_static`), which
keeps the consumers' `joy_stale` (0.2 s) watchdog quiet; if `hz` shows it publishes only on
motion, raise `joy_stale` in `joy_spacemouse.yaml` and `spacemouse_teleop.yaml`. Press the
side key you want as dead-man and read its index off `buttons`; put it in `deadman_button` of
both files (12 is the expected index of SHIFT, lower-left row, on the SpaceMouse Pro;
unverified until the driver runs). The dead-man is held, not toggled: `require_deadman`
turns the requirement on and off (sim only off).

### Collection with the SpaceMouse

```bash
ros2 launch movensys_imitation_ros guard.launch.py mode:=sim device:=spacemouse
ros2 launch movensys_imitation_ros collect.launch.py mode:=sim operator:=name device:=spacemouse
```

`device:=quest` (default) or `device:=spacemouse` appends `config/dobot_cr3a/joy_<device>.yaml`
after `guard.yaml` (topic, axes, deadzone, dead-man); with the SpaceMouse the driver is
included (`driver:=false` if `spacenav.launch.py` already runs in another shell). The guard log
line must read `joy /spacenav/joy axes [0, 1] deadman=button <n>`.

Calibrate once from where you sit or stand while collecting, exactly as for the Quest stick:
hold the dead-man, push the puck away from you, and the tip must move away from you; push it
right and the tip must move to your right. Fix `stick_yaw_deg` (90-degree steps first, then
fine) and `stick_invert_x` / `stick_invert_y` in `joy_spacemouse.yaml` until both agree. The
deadzone starts at 0.05 (the puck self-centres; the Quest needs 0.15) and `expo` 2.0; adjust
after the first episodes. `max_speed` is shared with the Quest in `guard.yaml`.

### Free motion: parking and reach checks

```bash
ros2 launch movensys_imitation_ros spacemouse_teleop.launch.py mode:=sim            # 6-DoF
ros2 launch movensys_imitation_ros spacemouse_teleop.launch.py mode:=sim motion_mode:=planar
ros2 topic echo /spacemouse_teleop/state
ros2 param set /spacemouse_teleop motion_mode translation
```

Parameters: `config/dobot_cr3a/spacemouse_teleop.yaml`. Each 50 Hz cycle maps the six axes
to a unitless `TwistStamped` in `world_manipulator` (or `Link6` with `frame_id`): per-axis
deadzone 0.05 and expo, the motion mask (`full`, `translation`, `rotation`, `planar`,
`vertical`, `yaw`; same presets as the Quest teleop), `yaw_deg` to rotate the operator's x, y
onto the robot base (translation and roll/pitch together), then `linear_gain` and
`angular_gain` as fractions of Servo's `scale` (0.4 m/s, 0.8 rad/s in
`movensys_manipulator_moveit_config` `servo.yaml`): the starting 0.05 gives 2 cm/s and
2.3 deg/s at full deflection; raise after the first runs. Calibrate signs with `axis_sign`
and `yaw_deg` the same way as the stick, one axis at a time.

States on `~/state`: `moving`, `hold:centred` (dead-man held, puck at rest: zero twists keep
Servo engaged), `hold:deadman` (released: five zero twists, then silence and Servo's own
0.1 s timeout halt), `hold:joy_stale` (driver gone: nothing published), `hold:pose_writer`
and `hold:servo_mode` (another owner, or the TWIST switch not yet accepted),
`fault:eef_stale` (no TF: the tool floor cannot be enforced, so nothing is published). The
floor zeroes downward translation once `Link6` z is at `table_z + link6_to_tip_bottom +
tip_floor_clearance` (tip on the table at 0.0; `z_floor_enabled: false` only without a
table), because Servo's collision world does not contain the table. Servo's singularity and
joint-limit stops apply as always. Optional keys: `mode_cycle_button` steps through
`mode_cycle_list`, `gripper_button` toggles `/wmx/set_gripper`.

Verified 2026-10-29 without Servo (topics remapped, a fake switch service and a fake Joy
with axis 1 at 1.0 and the dead-man held): one TWIST switch request, `moving`, 50 Hz
`TwistStamped` in `world_manipulator` with `linear.y` 0.05; with a fake publisher on the
pose topic the node holds with `hold:pose_writer` and never switches.

### Dry run without touching Servo

```bash
ros2 run movensys_imitation_ros spacemouse_teleop --ros-args \
  -r /servo_node/delta_twist_cmds:=/pusht/teleop_dryrun \
  -r /servo_node/switch_command_type:=/pusht/teleop_dryrun_switch \
  --params-file install/movensys_imitation_ros/share/movensys_imitation_ros/config/dobot_cr3a/task.yaml \
  --params-file install/movensys_imitation_ros/share/movensys_imitation_ros/config/dobot_cr3a/spacemouse_teleop.yaml
```

The guard dry run above takes `joy_spacemouse.yaml` as a third `--params-file`.

### Evidence for the done cell

Stick path, as for P2.2 with the SpaceMouse: `/pusht/target_guarded` at 50 Hz, z within 1 mm,
`hold:` at all four edges, `hold:deadman` on releasing the key, `hold:joy_stale` after
stopping `spacenav_node`, no `servo_halted` inside the area. Twist path: park the rod at push
height from the home pose with `spacemouse_teleop` alone, the floor holds the tip at the table
(`hold:z_floor` while pushing down), Servo halts within 0.2 s of releasing the key, and
`ros2 topic info -v` on both Servo input topics shows one writer at any time. Unit tests:
`pytest movensys_imitation_ros/test/test_spacemouse_core.py` and the dead-man cases in
`test_guard_core.py`.

## Scorer (P2.3)

Coverage is `area(T ∩ goal) / area(goal)` as in gym-pusht, success at 0.95; IoU is reported
next to it (the paper's metric). Two backends, both in
`movensys_imitation/tasks/pusht/scoring.py` (no ROS, unit-tested against hand-computed cases and
synthetic overhead frames):

* **geometric**: shapely on the block pose from `/pusht/t_block/pose` (sim ground truth);
* **pixel**: the T mask from the overhead crop (HSV threshold, 5 px opening, largest
  8-connected component) against a goal mask captured once with the T on the goal. Mask area
  within 15 % of the reference area means an unoccluded T; otherwise `pix_valid` is 0 and the
  coverage is a lower bound (the tip or wrist only removes T pixels), never a false success.

The pixel backend runs on the full-resolution crop (about 27 px per T unit at 1280 x 720 from
0.7 m), not on the 256 px policy image, where quantisation alone would exceed the tolerance.
The scale is taken from the T's own pixel area in the calibration frame, so no intrinsics are
needed and the real camera (P4.3) uses the same code and file format.

### Camera setup (once per camera pose)

With the scene playing and the arm parked away from the sheet, from the venv:

```bash
python -m movensys_imitation_ros.setup_camera --mode sim --camera-height 0.70
```

It teleports the T to `goal_pose()`, waits 2 s, checks the published pose (2 mm, 1 deg), grabs
one frame, finds the T with the `initial_threshold` of `config/dobot_cr3a/scorer.yaml`, fits a
square crop around the goal (sheet plus 3 cm margin), refines the threshold from the T's own
pixels (`refine` margins), and writes `config/dobot_cr3a/calibration/sim/`: `calibration.yaml`
(crop, scale, T area, threshold), `goal_mask.png`, `reference.png` and `goal_overlay.png` (the
mask edge in magenta and the expected outline in orange on the crop; look at it). Warnings
(extra blue components, scale off camera_info by more than 10 %, crop smaller than the sheet)
block the write unless `--force`. On the real robot: `--mode real`, place the T on the printed
outline by hand when prompted. The sim calibration is committed; the real one is produced at P4.3.

### Running

```bash
ros2 launch movensys_imitation_ros scorer.launch.py mode:=sim      # python -m from the venv
python -m movensys_imitation_ros.pusht_scorer --ros-args -p use_sim_time:=true \
  --params-file movensys_imitation_ros/config/dobot_cr3a/task.yaml \
  --params-file movensys_imitation_ros/config/dobot_cr3a/scorer.yaml
```

Outputs at 10 Hz: `/pusht/score` (`std_msgs/Float32MultiArray`, fields in the layout label:
`coverage, iou, success, coverage_pix, iou_pix, pix_valid, area_ratio, coverage_geom, iou_geom`;
the first three come from the geometric backend in sim and the pixel backend on the real robot),
`/pusht/t_mask` (mono8 crop-sized mask, for `rqt_image_view`), `/pusht_scorer/state` on change
(`ok`, `ok:occluded`, `ok:no_pose`, `hold:no_image`, `fault:no_calibration`). The log prints the
pixel-versus-geometric difference every 2 s and a summary at shutdown. The
`/pusht_scorer/score` Trigger service returns a fresh score as JSON; the episode manager calls it
after the arm retracts for the authoritative end-of-episode score.

### Evidence for the done cell

```bash
python -m movensys_imitation_ros.scorer_validation --out results/sim/scorer_validation.csv
```

Teleports the T to 20 poses (the goal, 10 within 3 mm and 3 deg of it, 5 at 5 to 30 mm, 4 random
starts), scores each with both backends and writes the CSV; PASS when every pose is valid and
within 0.02. Unit tests: `pytest movensys_imitation/tests/test_pusht_scoring.py`.

## Collection (P2.4)

Three venv nodes around the guard: `episode_recorder` samples the observation at exactly 10 Hz
into a LeRobot dataset, `episode_manager` runs the episode loop, and `pusht_scorer` (P2.3) gives
the live and the end-of-episode score. `collect.launch.py mode:=sim` starts all four; keys come
from `collect_keys` in another shell because launched processes have no terminal.

### Dataset

Plan Section 5 schema, as `movensys-convert-gym` writes it (`recorder_features`): the overhead
crop resized to 256 x 256 (`observation.images.top`, the scorer's calibration crop, so the policy
image and the pixel scorer share one camera setup), `observation.state` (Link6 x, y from TF, the
same source as the guard), `action` (the guard's `/pusht/target_guarded` at the tick),
`observation.joints`, `block_pose` (NaN on the real robot), `coverage` and `iou` (scorer fields
0 and 1). `observation.state` and `action` are stored in the sheet frame (planning frame minus
`task.area_center`), the frame of `block_pose`, the goal and the start marks, so the dataset
does not depend on where the sheet sits in front of the robot; the offset is in the sidecar as
`sheet_origin_wm` and the replay check adds it back. Timestamps are frame index / fps, so the
recorder ticks on the node clock (`use_sim_time` in sim) and keeps a timing verdict per
episode: a tick later than 0.1 s + `tick_tolerance` counts as missed, an image that arrived more
than `max_image_age` before the tick as stale (arrival on the node clock, not the header stamp:
Isaac's camera stamps restart independently of `/clock`); either makes `timing_ok` false. The
dataset is finalized after every episode, so it is readable mid-session and a crash loses at
most the episode being recorded. Episode metadata the LeRobot tables cannot hold (seed, start pose, home point, operator,
stop reason, end score, success, time to success, timing, versions) goes to
`<root>/meta/movensys_episodes.jsonl`, one line per saved episode (`recorder_core.read_sidecar`).
The dataset is resumed if `<root>/meta/info.json` exists, so sessions append.

### Episode loop

Per episode, with seed `seed_start + k` (collection seeds are 1000 and up):

1. **Reset**: guard source `pose` (the arm holds), `~/lift` up (`lift_height` 0.08 m above the
   latched plane, ramped at the step limit), home to the `task.yaml` `home_candidates` entry
   farthest from the T's start footprint (`manager_core.choose_home`, at least
   `min_home_clearance` from the outline), teleport the T to `sample_start_pose(seed)` (2 mm,
   1 deg, three tries, then 0.5 s at rest), lower, `lower_settle`.
2. **Record**: `episode_meta` set on the recorder, guard source back to `teleop_source`
   (`stick`), recorder `~/start`. Stops on coverage >= 0.95 (`/pusht/score` field 0), on
   `max_steps` (300 frames) or on `~/stop` / `~/discard`.
3. **End**: recorder `~/stop`, guard to `pose`, lift up, `/pusht_scorer/score` for the
   authoritative score (geometric in sim, pixel on the real robot: the lift clears the tip from
   the T), `manager_core.decide_outcome` (operator discard, timing fault with
   `discard_on_timing_fault`, failures kept unless `keep_failures` is false), recorder `~/save`
   or `~/discard`. The tool stays lifted in `pose` source until the next reset.

A fault in any step (timeout, refused service, teleport not reached) ends the run safely:
recorder discard, guard to `pose`, lift up, state `fault:<why>`; `~/start` runs the next seed.

### Running

```bash
source env.bash
colcon build --symlink-install --base-paths movensys_imitation_ros --packages-select movensys_imitation_ros
source env.bash
ros2 launch movensys_imitation_ros collect.launch.py mode:=sim operator:=name seed_start:=1000 n_episodes:=10
python -m movensys_imitation_ros.collect_keys      # s start, x stop+save, d discard, a abort, q
ros2 topic echo /episode_manager/state             # idle | reset:* | record:<n> | end:* | fault:*
```

Preconditions as for the guard, except the push height: in sim the launch sets the guard's
`latch_mode` to `nominal`, so the plane is table + nominal clearance and the tool ramps there
wherever it was parked (a restart after a fault starts with the tool lifted). On the real robot
`latch_mode` stays `measured`: park the rod at push height first. `quest_servo_teleop` must not
be running. The grip is needed only while recording: the manager drops `require_deadman` for
its own lift, home and lower moves and restores it before switching to the stick
(`scripted_deadman: true` keeps it throughout). Isaac runs at about 0.55 real time on the lab
box, and the recorder ticks on the sim clock, so a 300-frame episode takes about 55 s of wall
time. `auto_start:=true` runs `n_episodes` back to back after the first `s`. Parameters:
`config/dobot_cr3a/recorder.yaml`.

```bash
lerobot-dataset-viz --repo-id movensys/pusht_cr3a_sim_v1 --root datasets/movensys/pusht_cr3a_sim_v1 --episode-index 0
```

### Checks for the done cell

```bash
python -m movensys_imitation_ros.reset_check --n 10 --out results/sim/reset_check.csv
python -m movensys_imitation_ros.replay_check --root datasets/movensys/pusht_cr3a_sim_v1 --out results/sim/replay_check.csv
```

`reset_check` needs only the scene: 10 teleports, each within 2 mm and 1 deg, and no motion
(0.5 mm, 0.2 deg) between 0.5 s and 1.5 s after it. `replay_check` needs the guard and the
scorer (the collect launch can stay up; `ros2 param set /planar_guard require_deadman false`
first, sim only, or hold the grip): for every episode it homes the tip to
the first `observation.state`, teleports the T to the first `block_pose`, replays the recorded
`action` sequence as a timed path on `/pusht/waypoints` and compares the end state with the
recorded one. The plan's "reproduce the final T pose" is made numeric: coverage within 0.05,
junction within 10 mm, yaw within 5 deg. Exact agreement is not expected: the replayed path is
the 10 Hz target linearly interpolated, not the original 50 Hz stream, and PhysX contact is not
bit-reproducible. Unit tests: `pytest movensys_imitation_ros/test` (`test_recorder_core.py`,
`test_manager_core.py`, the lift ramp in `test_guard_core.py`).
