# Isaac Sim scenes (plan Phase 2, task P2.1)

`9a_pusht_simulation.usd` is the PushT scene for the simulated CR3A: movensys-simulation's
`3a_trajectory_simulation.usd` plus the target sheet, the T block, a pusher rod under `Link6`,
an oblique camera and the two OmniGraph ROS 2 graphs the collection and inference nodes rely on.
The 3a gripper body and finger visuals are hidden (they blocked the camera's view of the T).
It is generated, not hand-edited: `build_scene.py` rebuilds it from the base scene, so changes go
into the script (or its flags) and the scene is regenerated. GUI edits to the saved file are lost
on the next rebuild.

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
scored by segmenting the dark blue T in the overhead image, as in Chi et al. 2023, and the sheet
artwork is thin and coloured (green outline, light blue marks) so the T colour threshold ignores it.

## Regenerate

```bash
# 1. textures and sheet spec (project venv)
python sim/isaacsim/make_assets.py
python sim/isaacsim/make_usd_assets.py        # standalone T block and pusher tip USD (needs usd-core: pip install -e '.[sim]')

# 2. the scene (Isaac Sim's Python, system ROS 2 Jazzy sourced so the bridge loads)
source /opt/ros/jazzy/setup.bash
~/isaacsim/6.2/python.sh sim/isaacsim/build_scene.py              # ~2 min headless
~/isaacsim/6.2/python.sh sim/isaacsim/build_scene.py --play 60    # build, then play 60 s for checks

# 3. with the scene playing (GUI or --play), in the project venv
python sim/isaacsim/smoke_check.py --image /tmp/top.png

# 4. the camera moved, so the scorer's crop and goal mask must follow (scene playing, arm parked)
python -m movensys_imitation_ros.setup_camera --mode sim
```

`build_scene.py` still authors the T and the tip inline; the two files in `assets/` are the same
geometry, materials and physics as standalone USD (ASCII `.usd`, default prim set) for other
scenes. `t_block.usd`: origin at the bar-stem junction on the bottom face, rigid body, 0.15 kg.
`pusher_tip.usd`: origin at the mount face, cylinder along +z; under `Link6` place it at the
origin so the tip bottom is at 0.22 m. Reference them with
`prim.GetReferences().AddReference("./assets/t_block.usd")`.

`build_scene.py --help` lists the flags: `--center-wm X Y` (push-area centre in `world_manipulator`,
default `-0.03 -0.05`), `--jetson-xy X Y` (where the Jetson box goes, default `0.40 0.25`),
`--camera-sheet X Y Z` (camera position in the sheet frame, default `0.015 -0.35 0.40`),
`--camera-look-at X Y` (where the optical axis meets the table, default the sheet centre) and
`--tip-length` (0.22 m).

## Frames and numbers

* **Sheet frame** = prim `/World/pusht`: origin at the push-area centre on the table top, axes
  parallel to `world_manipulator` (the robot base is on the +y side). Everything PushT-specific is
  expressed here: the block pose topic, the teleport topic, the goal pose and the start marks.
* **Push area**: 0.60 m wide (x) by 0.30 m deep (y) (plan v2.5; a 0.45 m square until 2026-10-24,
  which the arm could not reach at either edge). With the default centre (-0.03, -0.05) the sheet
  spans `world_manipulator` x -0.33..0.27, y -0.20..0.10; the table is x -0.50..0.50, y -0.35..0.35
  and the base is at y = 0.275. The near edge and the far corners are inside Servo's singularity
  slow-down zone, so teleop is slower there. The 3a scene's Jetson box (0.20 x 0.20 m, 3a default
  (0.36, 0.18)) is parked in the table corner at (0.40, 0.25) by `--jetson-xy`; the manipulator
  URDF still places it at (0.36, 0.18), so MoveIt's collision world disagrees with Isaac there
  until `movensys_manipulator.urdf` follows.
* **Table top**: z = 0.0431 m in `world_manipulator` (measured from the 3a collision mesh).
* **Pusher tip**: 30 mm rod on the `Link6` +z axis from the flange (z = 0) to 0.22 m. The tip
  bottom is 0.22 m from the `Link6` origin, so the guard's z setpoint for a tip 10 mm above the
  table is **0.2731 m** (in `scene_9a.json`). Until 2026-11-05 the rod ran from 0.14 m, below
  the gripper body; the bottom did not move, so `task.yaml` and the guard are unchanged.
* **Hidden robot parts**: `Link6/bracket/gripper` (the gripper body, a visual-only prim of the
  robot payload) is deactivated and `picker_1/visuals`, `picker_2/visuals` are invisible. The
  finger bodies, their prismatic joints, `/joint_states` and the gripper service are unchanged.
  MoveIt's collision world (URDF) still has the gripper.
* **T block**: gym-pusht proportions at 30 mm per unit (v2.4), 30 mm tall, 0.15 kg, friction 0.40/0.35,
  dark blue (20, 40, 110). Its pose origin is the bar-stem junction; yaw 0 puts
  the bar along +x on the +y side. It starts at start-state seed 0.
* **Camera**: a plain `Camera` prim (`/World/camera_top_link/camera_top`) with the D455 colour
  optics (focal 1.93 mm, aperture 3.896 mm: 90.5 degree horizontal field of view, fx 634 px),
  1280 x 720 at 30 Hz. Since 2026-11-05 it is an **oblique view**: in the sheet frame the camera
  sits at (0.015, -0.35), 5 cm beyond the operator-side table edge, 0.40 m above the table, and
  its optical axis goes through the sheet centre (41 degrees from vertical). The straight-down
  view from 0.70 m lost the T behind the forearm during most of each push (measured 2026-11-03).
  Image up is the sheet +y direction projected into the image, so the robot base is at the top
  of the picture and image +x is sheet +x. Nothing of the D455 model is in the scene any more.
  Pixel scale is no longer uniform: about 1400 px/m at the near sheet edge and 1000 px/m at the
  far edge, so the pixel scorer is only exact at the goal, where its mask was captured
  (`setup_camera` writes the crop and goal mask for this view).

## ROS 2 interfaces (Appendix A of the plan)

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/joint_states`, `/joint_command`, `/clock` | unchanged from 3a | | |
| `/image_top/rgb`, `/image_top/camera_info` | `sensor_msgs/Image` (rgb8), `CameraInfo` | out, 30 Hz | frame `camera_top_color_optical_frame`; oblique view since 2026-11-05 |
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
