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
ros2 service call /wmx/moveit2/joint_movement \
  movensys_manipulator_moveit_config/srv/MoveJoints \
  "{joint_names: [joint1, joint2, joint3, joint4, joint5, joint6],
    joint_values: [0.1511, 0.0239, -1.8702, 0.2709, 1.5685, 0.1512]}"


or 

ros2 service call /wmx/moveit2/absolute_base_eef_joint_movement \
  movensys_manipulator_moveit_config/srv/MovePose \
  "{pos: [-0.107, -0.071, 0.284], ori: [3.137, -0.003, -3.141]}"


## Episode collection

ros2 launch movensys_imitation_ros collect.launch.py mode:=sim operator:=mali n_episodes:=1

python -m movensys_imitation_ros.collect_keys



## Isaac Teleoperation
```
mtele ros2 run quest_pose_publisher quest_pose_node --ros-args \
    -p hand:=both -p pose_source:=aim -p cloudxr_accept_eula:=true
```

Controller Web ui will be available at this location:
https://192.168.0.20:48322/client/?codec=h264&maxStreamingBitrateMbps=80&perEyeWidth=1024&perEyeHeight=1024&deviceFrameRate=72#/real/gear/headless
