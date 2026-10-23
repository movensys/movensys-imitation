## Start Isaac Simulation



## Isaac Sim Bridge
mros ros2 launch movensys_manipulator_moveit_config sim_bridge.launch.py simulator:=isaacsim use_sim_time:=true

## Start Moveit2
mros ros2 launch movensys_manipulator_moveit_config moveit.launch.py use_sim_time:=true


## Start Moveit Servo
mros ros2 launch movensys_manipulator_moveit_config quest_servo_teleop.launch.py use_sim_time:=true


## Isaac Teleoperation
```
mtele ros2 run quest_pose_publisher quest_pose_node --ros-args \
    -p hand:=both -p pose_source:=aim -p cloudxr_accept_eula:=true
```

Controller Web ui will be available at this location:
https://192.168.0.20:48322/client/?codec=h264&maxStreamingBitrateMbps=80&perEyeWidth=1024&perEyeHeight=1024&deviceFrameRate=72#/real/gear/headless
