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
