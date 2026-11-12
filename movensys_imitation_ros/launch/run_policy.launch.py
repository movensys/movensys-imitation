"""Launch autonomous policy episodes in sim (plan P2.8).

Guard, scorer, recorder, manager and policy_runner. Same preconditions as collect.launch.py
(scene playing, sim_bridge and MoveIt up, no teleop node). The manager runs its usual loop
with ``teleop_source pose``; the policy runner fills the recording phase. Episodes are
recorded like demonstrations into a separate dataset (``dataset`` argument), whose sidecar
carries success and coverage per episode. ``s`` in
``collect_keys`` starts; ``auto_start:=true`` runs ``n_episodes`` back to back.

    ros2 launch movensys_imitation_ros run_policy.launch.py mode:=sim \
        pretrained:=outputs/sim/diffusion_v3_s1/checkpoints/last/pretrained_model \
        overrides:="--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10" \
        run_mode:=slow dataset:=eval_diffusion_v3_s1 seed_start:=2000 n_episodes:=10 \
        auto_start:=true

``run_mode:=shadow`` with ``source:=stick`` logs the policy output on ``/pusht/policy_target``
while an operator drives.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from movensys_imitation_ros.launch_support import perform, sim_time_and_mode, venv_process


def launch_setup(context):
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    task_yaml = os.path.join(config, "task.yaml")
    common = sim_time_and_mode(context)
    dataset = perform(context, LaunchConfiguration("dataset"))
    return [
        venv_process(
            context, "pusht_scorer", [task_yaml, os.path.join(config, "scorer.yaml")], common
        ),
        venv_process(
            context, "episode_recorder", [task_yaml, os.path.join(config, "recorder.yaml")],
            {
                **common,
                "repo_id": f"movensys/{dataset}",
                "root": f"datasets/movensys/{dataset}",
            },
        ),
        venv_process(
            context, "episode_manager", [task_yaml, os.path.join(config, "recorder.yaml")],
            {
                **common,
                "operator": LaunchConfiguration("operator"),
                "teleop_source": LaunchConfiguration("source"),
                "input_device": "policy",
                "seed_start": LaunchConfiguration("seed_start"),
                "n_episodes": LaunchConfiguration("n_episodes"),
                "auto_start": LaunchConfiguration("auto_start"),
            },
        ),
        venv_process(
            context, "policy_runner", [task_yaml, os.path.join(config, "runner.yaml")],
            {
                **common,
                "pretrained": LaunchConfiguration("pretrained"),
                "policy_overrides": LaunchConfiguration("overrides"),
                "run_mode": LaunchConfiguration("run_mode"),
                "trigger": LaunchConfiguration("trigger"),
            },
        ),
    ]


def generate_launch_description():
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    mode = LaunchConfiguration("mode")
    use_sim_time = PythonExpression(["'", mode, "' != 'real'"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="sim", description="sim | hil | real"),
            DeclareLaunchArgument(
                "pretrained", description="checkpoint pretrained_model directory"
            ),
            DeclareLaunchArgument(
                "overrides", default_value="",
                description="space-separated policy overrides, e.g. --policy.fm_num_steps=4",
            ),
            DeclareLaunchArgument(
                "run_mode", default_value="slow", description="shadow | slow | normal"
            ),
            DeclareLaunchArgument(
                "trigger", default_value="recorder", description="recorder | always"
            ),
            DeclareLaunchArgument(
                "source", default_value="pose",
                description="guard source while recording: pose (policy drives) | stick (shadow)",
            ),
            DeclareLaunchArgument(
                "dataset", default_value="eval_policy",
                description="dataset name under datasets/movensys/",
            ),
            DeclareLaunchArgument(
                "operator", default_value="policy", description="sidecar metadata"
            ),
            DeclareLaunchArgument(
                "seed_start", default_value="2000", description="first start-pose seed"
            ),
            DeclareLaunchArgument("n_episodes", default_value="10"),
            DeclareLaunchArgument("auto_start", default_value="true"),
            Node(
                package="movensys_imitation_ros",
                executable="planar_guard",
                name="planar_guard",
                output="screen",
                parameters=[
                    os.path.join(config, "task.yaml"),
                    os.path.join(config, "guard.yaml"),
                    os.path.join(config, "joy_spacemouse.yaml"),
                    {
                        "use_sim_time": use_sim_time,
                        "source": "pose",
                        "latch_mode": PythonExpression(
                            ["'nominal' if '", mode, "' == 'sim' else 'measured'"]
                        ),
                    },
                ],
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
