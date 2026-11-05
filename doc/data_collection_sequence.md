## Start Isaac Simulation


## Isaac Sim Bridge
mros ros2 launch movensys_manipulator_moveit_config sim_bridge.launch.py simulator:=isaacsim use_sim_time:=true

## Start Moveit2
mros ros2 launch movensys_manipulator_moveit_config moveit.launch.py use_sim_time:=true

## Close the gripper
source /opt/ros/jazzy/setup.bash
ros2 service call /wmx/set_gripper std_srvs/srv/SetBool "{data: true}"    # close

ros2 service call /wmx/set_gripper std_srvs/srv/SetBool "{data: false}"   # open

## Park rod at push height 

With the SpaceMouse (stop it with Ctrl-C before the collection launch; it is a Servo writer):
```
ros2 launch movensys_imitation_ros spacemouse_teleop.launch.py mode:=sim
```
Hold the dead-man key and drive the tip to about 10 mm above the sheet, tool vertical
(`motion_mode:=translation` keeps the orientation). Or by service call from inside the
manipulator container (`mros` opens a shell there; the `movensys_manipulator_moveit_config`
service types are not installed on the host, so the host shell answers "The passed service type
is invalid"):


```
mros
ros2 service call /wmx/moveit2/absolute_base_eef_joint_movement \
  movensys_manipulator_moveit_config/srv/MovePose \
  "{pos: [-0.107, -0.071, 0.284], ori: [3.137, -0.003, -3.141]}"
```


## Episode collection


With the SpaceMouse as the stick (starts spacenav_node too; hold the dead-man key to move):

ros2 launch movensys_imitation_ros collect.launch.py mode:=sim operator:=mali n_episodes:=1 device:=spacemouse

python -m movensys_imitation_ros.collect_keys



## Isaac Teleoperation
```
mtele ros2 run quest_pose_publisher quest_pose_node --ros-args \
    -p hand:=both -p pose_source:=aim -p cloudxr_accept_eula:=true
```

Controller Web ui will be available at this location:
https://192.168.0.20:48322/client/?codec=h264&maxStreamingBitrateMbps=80&perEyeWidth=1024&perEyeHeight=1024&deviceFrameRate=72#/real/gear/headless

## SpaceMouse teleoperation (3Dconnexion)
```
ros2 launch movensys_imitation_ros guard.launch.py mode:=sim device:=spacemouse driver:=false
```

`require_deadman` is not a launch argument (`ros2 launch` takes only `name:=value` pairs, no
`--ros-args -p`). To drop the dead-man grip in sim, set it at runtime from a second terminal
once the guard is up:
```
ros2 param set /planar_guard require_deadman false
```


## Camera feed
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view /image_top/rgb

rqt 
and load default perscpective

The scene has one camera since 2026-11-05: the oblique top view (`/image_top/rgb`). The wrist
camera and the gripper were removed because they hid the T.

python3 - <<'EOF'
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
rclpy.init()
n = Node("coverage_watch")
n.create_subscription(Float32MultiArray, "/pusht/score",
    lambda m: print(f"coverage {m.data[0]:.3f}  success {int(m.data[2])}", flush=True), 10)
rclpy.spin(n)
EOF


ros2 launch movensys_imitation_ros scorer.launch.py mode:=sim






## Replay a saved episode

Two ways. The dataset holds the top camera (`observation.images.top`), plus state, action,
joints, T pose and coverage. The top camera moved to an oblique view on 2026-11-05 (decided
2026-11-03, after measuring that the straight-down view loses the T behind the forearm for most
of each push). Episodes from the old camera live in `pusht_cr3a_sim_v1`; everything recorded
with the oblique camera goes to `pusht_cr3a_sim_v2` (the recorder default), never to the old root.

### A. Offline viewer: what was recorded

```
HF_HUB_OFFLINE=1 lerobot-dataset-viz --repo-id movensys/pusht_cr3a_sim_v2 \
  --root datasets/movensys/pusht_cr3a_sim_v2 --episode-index 0
```

Opens a rerun window: the top-camera video, `observation.state` (tip x, y), `action`, the joints,
`block_pose` and `coverage` per frame, with a time cursor. `--save 1 --output-dir <dir>` writes
an `.rrd` file instead of opening the window. Episode indices are in
`datasets/movensys/pusht_cr3a_sim_v2/meta/movensys_episodes.jsonl` (one line per episode, with
seed, stop reason, success and end score).

### B. Live replay in Isaac Sim: the robot re-executes the recorded actions

Needs the scene playing, the guard and the scorer. The collection launch can stay up while the
manager is idle (do not press `s` during a replay). Drop the dead-man first (sim only):

```
ros2 param set /planar_guard require_deadman false
```

Open the camera feed and keep RViz in the manipulator container for the arm:

```
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view /image_top/rgb
```

Replay one episode from the venv shell with the workspace sourced:

```
source install/setup.bash
python -m movensys_imitation_ros.replay_check --episodes 0 --out /tmp/replay_0.csv
```

Add `--root <other dataset>` for a dataset other than the default `pusht_cr3a_sim_v2` (the old
`pusht_cr3a_sim_v1` episodes replay fine, only their recorded images are from the old camera).

```
```

For each episode it lifts, homes the tip to the episode's first `observation.state`, teleports
the T to the first `block_pose`, lowers, publishes the recorded 10 Hz actions as a timed path on
`/pusht/waypoints`, lifts and scores. It prints the position, yaw and coverage differences
against the recording (PASS within 10 mm, 5 deg, 0.05) and writes them to the CSV. Omit
`--episodes` to replay the whole dataset; each episode takes its recorded length plus about
10 s of reset. Exact agreement is not expected: the path is the 10 Hz target interpolated, and
PhysX contact is not bit-reproducible.
