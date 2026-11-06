"""
policy_runner: closed-loop policy execution through the planar guard (plan P2.8).

Subscribes
    /image_top/rgb            sensor_msgs/Image         overhead frame (crop from the calibration)
    /episode_recorder/state   std_msgs/String           gate: runs while "recording:<n>"
    TF world_manipulator -> Link6 (or /wmx/moveit2/eef_pose with eef_source topic)
Publishes
    /pusht/target_raw         geometry_msgs/PoseStamped guard target (modes slow, normal)
    /pusht/policy_target      geometry_msgs/PoseStamped the raw policy output (all modes)
    ~/state                   std_msgs/String           idle | running:<n> | stale | fault:<why>

Every tick (``task.rates.policy_hz``, the recorder's rate, on the node clock) the current frame
is cropped and resized exactly as the recorder does, the tip position is shifted into the sheet
frame, ``PolicyRunner.act`` gives the next target in the sheet frame and it is published in the
planning frame. The episode manager owns the episode (reset, lift, scoring, saving): launch it
with ``teleop_source pose`` and this node fills the recording phase, like an operator would.

Modes: ``shadow`` only publishes ``/pusht/policy_target`` (an operator drives, the prediction is
logged next to the action); ``slow`` limits each target to ``slow_step`` from the tip;
``normal`` publishes the policy output. The guard's own limits (5 mm per 50 Hz cycle, 30 mm lead,
area clamp) apply in every mode. Runs in the venv (``python -m``), like the recorder.
"""

from __future__ import annotations

import os
from pathlib import Path

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tf2_ros

from movensys_imitation.tasks.pusht import scoring as sc
from movensys_imitation_ros import recorder_core as rc
from movensys_imitation_ros import runner_core as rn
from movensys_imitation_ros.pusht_scorer import default_calibration_dir


def resolve_path(p: str) -> Path:
    path = Path(os.path.expanduser(p))
    if path.is_absolute():
        return path
    root = os.environ.get("MOVENSYS_IMITATION_ROOT")
    return (Path(root) / path) if root else path.resolve()


class PolicyRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_runner")
        p = self._declare_parameters()
        if p["run_mode"] not in rn.RUN_MODES:
            raise ValueError(f"run_mode must be one of {rn.RUN_MODES}, got {p['run_mode']!r}")
        self.run_mode: str = p["run_mode"]
        self.slow_step: float = float(p["slow_step"])
        self.area_size = tuple(float(v) for v in p["task.area_size"])
        self.area_margin: float = float(p["area_margin"])
        self.origin = tuple(float(v) for v in p["task.area_center"])
        self.planning_frame: str = p["task.planning_frame"]
        self.eef_frame: str = p["task.eef_frame"]
        self.eef_source: str = p["eef_source"]
        self.max_image_age: float = float(p["max_image_age"])
        self.image_size: int = int(p["image_size"])
        self.gate = rn.EpisodeGate(p["trigger"])

        cal_dir = Path(p["calibration_dir"]) if p["calibration_dir"] else None
        cal_dir = cal_dir or default_calibration_dir(p["mode"])
        self.cal = sc.Calibration.load(cal_dir)

        pretrained = resolve_path(p["pretrained"])
        if not pretrained.exists():
            raise FileNotFoundError(f"pretrained checkpoint {pretrained} not found")
        overrides = tuple(s for s in p["policy_overrides"] if s)
        from movensys_imitation.inference import PolicyRunner  # torch import after rclpy init

        self.runner = PolicyRunner(pretrained, overrides, device=p["device"] or None)

        self.image: Image | None = None
        self.image_rx = -1.0
        self.eef_pose: PoseStamped | None = None
        self.recorder_state = ""
        self.n_ticks = 0
        self.stale_warned = False

        if self.eef_source == "topic":
            self.tf_buffer = None
            self.create_subscription(PoseStamped, p["eef_pose_topic"], self._on_eef_pose, 10)
        else:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, p["image_topic"], self._on_image, qos_profile_sensor_data)
        self.create_subscription(String, p["recorder_state_topic"], self._on_recorder, 10)
        self.target_pub = self.create_publisher(PoseStamped, p["target_topic"], 10)
        self.policy_pub = self.create_publisher(PoseStamped, "/pusht/policy_target", 10)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.state = ""
        self.create_timer(1.0 / float(p["task.rates.policy_hz"]), self._tick)
        self.create_timer(1.0, self._heartbeat)
        self._set_state("idle")
        self.get_logger().info(
            f"policy_runner: {self.runner.cfg.type} from {pretrained} "
            f"({', '.join(overrides) or 'no overrides'}), mode {self.run_mode}, "
            f"trigger {self.gate.trigger}, device {self.runner.cfg.device}, "
            f"{self.image_size} px crop from {cal_dir}, eef via {self.eef_source}, "
            f"sheet origin {self.origin}"
        )

    def _declare_parameters(self) -> dict:
        defaults = {
            "mode": "sim",
            "pretrained": "",
            "policy_overrides": [""],
            "device": "",
            "run_mode": "slow",
            "slow_step": 0.010,          # m per tick in slow mode (0.1 m/s at 10 Hz)
            "trigger": "recorder",       # recorder: while the recorder records; always
            "area_margin": 0.0,          # m kept from the area edge (0: guard clamp only)
            "image_size": 256,
            "calibration_dir": "",
            "max_image_age": 0.3,
            "image_topic": "/image_top/rgb",
            "target_topic": "/pusht/target_raw",
            "recorder_state_topic": "/episode_recorder/state",
            "eef_source": "tf",
            "eef_pose_topic": "/wmx/moveit2/eef_pose",
            "task.planning_frame": "world_manipulator",
            "task.eef_frame": "Link6",
            "task.area_center": [-0.03, -0.05],
            "task.area_size": [0.60, 0.30],
            "task.rates.policy_hz": 10.0,
        }
        return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

    # ------------------------------------------------------------------ callbacks

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_image(self, msg: Image) -> None:
        self.image = msg
        self.image_rx = self._now()

    def _on_eef_pose(self, msg: PoseStamped) -> None:
        self.eef_pose = msg

    def _on_recorder(self, msg: String) -> None:
        self.recorder_state = msg.data

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_pub.publish(String(data=state))

    def _heartbeat(self) -> None:
        self.state_pub.publish(String(data=self.state))

    def _eef_xy(self) -> tuple[float, float] | None:
        if self.eef_source == "topic":
            if self.eef_pose is None:
                return None
            return (self.eef_pose.pose.position.x, self.eef_pose.pose.position.y)
        try:
            tf = self.tf_buffer.lookup_transform(self.planning_frame, self.eef_frame, Time())
        except tf2_ros.TransformException:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    def _frame(self) -> np.ndarray | None:
        if self.image is None or self._now() - self.image_rx > self.max_image_age:
            return None
        rgb = sc.decode_image(
            self.image.data, self.image.width, self.image.height, self.image.step,
            self.image.encoding,
        )
        return rc.resize_crop(self.cal.apply_crop(rgb), self.image_size)

    def _publish(self, pub, xy_planning) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.planning_frame
        msg.pose.position.x, msg.pose.position.y = float(xy_planning[0]), float(xy_planning[1])
        msg.pose.orientation.w = 1.0
        pub.publish(msg)

    # ------------------------------------------------------------------ tick

    def _tick(self) -> None:
        event = self.gate.update(self.recorder_state)
        if event is None:
            return
        if event == "stop":
            self.get_logger().info(f"episode done after {self.n_ticks} policy ticks")
            self._set_state("idle")
            return
        if event == "start":
            self.runner.reset()
            self.n_ticks = 0
            self.stale_warned = False
            self.get_logger().info(f"episode start: policy history reset ({self.run_mode})")

        frame = self._frame()
        eef = self._eef_xy()
        if frame is None or eef is None:
            self._set_state("stale")
            if not self.stale_warned:
                self.stale_warned = True
                self.get_logger().warning(
                    f"skipping ticks: image {'missing' if frame is None else 'ok'}, "
                    f"eef {'missing' if eef is None else 'ok'}"
                )
            return
        state = rn.to_sheet(eef, self.origin)
        try:
            action = self.runner.act(frame, state)
        except Exception as e:  # noqa: BLE001  keep the node alive, report once per episode
            self._set_state(f"fault:{e}")
            self.get_logger().error(f"policy call failed: {e}", throttle_duration_sec=5.0)
            return
        if not rn.finite_xy(action):
            self._set_state("fault:nan")
            self.get_logger().error(
                "policy returned a non-finite action", throttle_duration_sec=5.0
            )
            return
        target = rn.clamp_area(action, self.area_size, self.area_margin)
        self._publish(self.policy_pub, rn.to_planning(target, self.origin))
        if self.run_mode == "slow":
            target = rn.limit_step(state, target, self.slow_step)
        if self.run_mode in ("slow", "normal"):
            self._publish(self.target_pub, rn.to_planning(target, self.origin))
        self.n_ticks += 1
        self._set_state(f"running:{self.n_ticks}")
        if self.n_ticks % 50 == 1:
            self.get_logger().info(
                f"tick {self.n_ticks}: tip ({state[0] * 1000:.0f}, {state[1] * 1000:.0f}) mm -> "
                f"target ({target[0] * 1000:.0f}, {target[1] * 1000:.0f}) mm, "
                f"{self.runner.last_latency_ms:.1f} ms"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyRunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
