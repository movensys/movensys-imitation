# Isaac Sim scenes (plan Phase 2, task P2.1)

`9a_pusht_simulation.usd` is the PushT scene for the simulated CR3A: movensys-simulation's
`3a_trajectory_simulation.usd` plus the target sheet, the T block, a pusher tip under `Link6`,
an overhead camera and the two OmniGraph ROS 2 graphs the collection and inference nodes rely on.
It is generated, not hand-edited: `build_scene.py` rebuilds it from the base scene, so changes go
into the script (or its flags) and the scene is regenerated.

## Files

| File | What |
|---|---|
| `9a_pusht_simulation.usd` | the scene (open in Isaac Sim with the ROS 2 bridge enabled, press Play) |
| `scene_9a.json` | numbers the ROS side needs: sheet frame pose, table height, tip offset, EEF z setpoint, camera, topics |
| `build_scene.py` | builds the scene with Isaac Sim's Python (see below) |
| `make_assets.py` | draws `assets/target_sheet.png` and writes `assets/pusht_sheet.json` |
| `make_usd_assets.py` | writes `assets/t_block.usd` and `assets/pusher_tip.usd`, standalone referenceable copies of the T and the tip |
| `smoke_check.py` | ROS 2 check of the running scene: topic rates, image format, teleport round trip |
| `assets/` | the sheet texture, the sheet spec, `t_block.usd`, `pusher_tip.usd`, `overhead_sample.png` (a downscaled `/image_top/rgb` frame) |

The task geometry (T proportions, goal pose, start-state sampler) lives in
`movensys_imitation/tasks/pusht/geometry.py` and is the single source for the artwork, the scene
and the scorer. There are no fiducial markers on the T or the sheet (plan v2.1): real episodes are
scored by segmenting the grey T in the overhead image, as in Chi et al. 2023, and the sheet
artwork is thin and coloured (green outline, blue marks) so a grey threshold ignores it.

## Regenerate

```bash
# 1. textures and sheet spec (project venv)
python sim/isaacsim/make_assets.py
python sim/isaacsim/make_usd_assets.py        # standalone T block and pusher tip USD (needs usd-core: pip install -e '.[sim]')

# 2. the scene (Isaac Sim's Python, system ROS 2 Jazzy sourced so the bridge loads)
source /opt/ros/jazzy/setup.bash
~/isaac-sim/python.sh sim/isaacsim/build_scene.py                 # ~2 min headless
~/isaac-sim/python.sh sim/isaacsim/build_scene.py --play 60       # build, then play 60 s for checks

# 3. with the scene playing (GUI or --play), in the project venv
python sim/isaacsim/smoke_check.py --image /tmp/top.png
```

`build_scene.py` still authors the T and the tip inline; the two files in `assets/` are the same
geometry, materials and physics as standalone USD (ASCII `.usd`, default prim set) for other
scenes. `t_block.usd`: origin at the bar-stem junction on the bottom face, rigid body, 0.25 kg.
`pusher_tip.usd`: origin at the mount face, cylinder along +z; under `Link6` place it at
z = 0.14 so the tip bottom is at 0.22 m. Reference them with
`prim.GetReferences().AddReference("./assets/t_block.usd")`.

`build_scene.py --help` lists the flags: `--center-wm X Y` (push-area centre in `world_manipulator`,
default `-0.03 -0.08`, to be fixed after the reach check), `--camera-height` (0.70 m) and
`--tip-length` (0.08 m).

## Frames and numbers

* **Sheet frame** = prim `/World/pusht`: origin at the push-area centre on the table top, axes
  parallel to `world_manipulator` (the robot base is on the +y side). Everything PushT-specific is
  expressed here: the block pose topic, the teleport topic, the goal pose and the start marks.
* **Push area**: 0.45 x 0.45 m. With the default centre the sheet spans `world_manipulator`
  x -0.255..0.195, y -0.305..0.145; the table edge is at y = -0.35 and the base at y = 0.275.
* **Table top**: z = 0.0431 m in `world_manipulator` (measured from the 3a collision mesh).
* **Pusher tip**: 20 mm cylinder on the `Link6` +z axis from 0.14 to 0.22 m. The tip bottom is
  0.22 m from the `Link6` origin, so the guard's z setpoint for a tip 10 mm above the table is
  **0.2731 m** (in `scene_9a.json`).
* **T block**: gym-pusht proportions at 40 mm per unit, 30 mm tall, 0.25 kg, friction 0.40/0.35,
  plain grey (120, 120, 120) like gym-pusht. Its pose origin is the bar-stem junction; yaw 0 puts
  the bar along +x on the +y side. It starts at start-state seed 0.
* **Camera**: a plain `Camera` prim (`/World/camera_top_link/camera_top`) with the D455 colour
  optics, 0.70 m above the table, optical axis straight down, 1280 x 720 at 30 Hz. The RealSense
  model under the same link is a visual prop only: its own camera prims are marked invisible and a
  render product on one of them shows the dome backdrop instead of the scene. Image +x is sheet +x
  and image up is sheet +y, so the robot is at the top of the picture. Horizontal field of view
  about 90 degrees (1.4 x 0.8 m on the table), so the sheet is about 410 x 410 px and the T bar
  about 145 x 36 px.

## ROS 2 interfaces (Appendix A of the plan)

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/joint_states`, `/joint_command`, `/clock` | unchanged from 3a | | |
| `/image_top/rgb`, `/image_top/camera_info` | `sensor_msgs/Image` (rgb8), `CameraInfo` | out, 30 Hz | frame `camera_top_color_optical_frame` |
| `/pusht/t_block/pose` | `geometry_msgs/Pose` | out, 30 Hz | block pose in the sheet frame |
| `/pusht/t_block/pose_cmd` | `geometry_msgs/Pose` | in | teleports the block and zeroes its velocity; send z = 0.001 |

All graphs are stock nodes. OmniGraph's vector split node does not accept quaternions (the
`7x` scenes' piece-pose graphs publish uninitialised orientations for that reason), so the pose
graph takes the Euler z angle of the block's local orient with `GetMatrix4Rotation` and publishes
`(0, 0, sin(yaw/2), cos(yaw/2))`, which is exact for a block that only yaws. The generic
`ROS2Publisher`/`ROS2Subscriber` nodes get their message type authored in USD after creation,
since setting it through the graph controller alone is lost when the stage is saved. Do not use
`ScriptNode` here: the headless Python app does not load that extension and one unknown node type
silently stops every graph in the stage, including the joint-state and clock graphs.

## Not done here

* The **reach check** (P2.1 "done when"): drive the tip over the whole sheet at the push height
  with Quest teleop and confirm no Servo singularity slow-down at the four corners, then fix
  `--center-wm` and regenerate. The default centre puts the far corners about 0.64 m from the
  base axis, which is at the edge of the CR3A's reach, so the centre may need to move toward the
  robot or the area may need to shrink.
* `9b_pusht_hil.usd` (Phase 3): the same build with the joint topics under `/isaacsim`.
