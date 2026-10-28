"""
episode_recorder: samples the PushT observation at exactly ``fps`` Hz into a LeRobotDataset.

Plan P2.4. Runs from the project venv (``python -m movensys_imitation_ros.episode_recorder``):
it needs LeRobot, numpy 2 and OpenCV. The logic is in :mod:`recorder_core`; this node only
collects the latest messages, ticks on the node clock (``use_sim_time`` in sim, so a slow
simulator still gives 10 Hz of sim time) and exposes the services the episode manager calls.

Inputs
    /image_top/rgb            sensor_msgs/Image         overhead frame (crop from the calibration)
    TF planning -> eef, or /wmx/moveit2/eef_pose        observation.state (x, y)
    /pusht/target_guarded     geometry_msgs/PoseStamped action (x, y), the guard's target
    /joint_states             sensor_msgs/JointState    observation.joints
    /pusht/t_block/pose       geometry_msgs/Pose        block_pose (sim only; NaN otherwise)
    /pusht/score              std_msgs/Float32MultiArray coverage, iou (fields 0 and 1)
Services (std_srvs/Trigger)
    ~/start     begin buffering on the next tick (fails while an input is missing)
    ~/stop      stop buffering, keep the frames
    ~/save      stop and write the episode; the response message is JSON (episode_index, timing)
    ~/discard   stop and drop the frames
Parameter ``episode_meta`` (JSON string) is merged into the sidecar line at save time; the
manager sets it before ``~/start`` (seed, operator) and again before ``~/save`` (end score).
Outputs
    ~/state     std_msgs/String   idle | recording:<n> | stopped:<n> | fault:<reason>
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from geometry_msgs.msg import Pose, PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger
import tf2_ros

from movensys_imitation.tasks.pusht import scoring as sc
from movensys_imitation_ros import recorder_core as rc
from movensys_imitation_ros.pusht_scorer import default_calibration_dir, pose_to_tuple


def resolve_root(root: str) -> Path:
    """Relative dataset roots are taken from the repo root (env.bash exports it), else the cwd."""
    p = Path(root).expanduser()
    if p.is_absolute():
        return p
    base = os.environ.get("MOVENSYS_IMITATION_ROOT")
    return (Path(base) if base else Path.cwd()) / p


class EpisodeRecorder(Node):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__("episode_recorder")
        p = self._declare_parameters()
        self.cfg = rc.RecorderConfig(
            repo_id=p["repo_id"],
            root=resolve_root(p["root"]),
            fps=int(p["fps"]),
            image_size=int(p["image_size"]),
            robot_type=p["robot_type"],
            joint_names=tuple(p["joint_names"]),
            max_image_age=p["max_image_age"],
            tick_tolerance=p["tick_tolerance"],
        )
        self.planning_frame: str = p["task.planning_frame"]
        self.eef_frame: str = p["task.eef_frame"]
        self.eef_source: str = p["eef_source"]
        cal_dir = Path(p["calibration_dir"]) if p["calibration_dir"] else None
        cal_dir = cal_dir or default_calibration_dir(p["mode"])
        self.cal = sc.Calibration.load(cal_dir)
        self.store = rc.EpisodeStore(self.cfg)

        self.image: Image | None = None
        self.image_rx = -math.inf
        self.eef_pose: PoseStamped | None = None
        self.action: tuple[float, float] | None = None
        self.action_rx = -math.inf
        self.joints: tuple[float, ...] | None = None
        self.joint_warned = False
        self.block_pose: tuple[float, float, float] | None = None
        self.coverage = math.nan
        self.iou = math.nan

        self.buffer: rc.EpisodeBuffer | None = None
        self.recording = False
        self.state = ""

        self.create_subscription(Image, p["image_topic"], self._on_image, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, p["action_topic"], self._on_action, 10)
        self.create_subscription(JointState, p["joint_topic"], self._on_joints, 10)
        self.create_subscription(Pose, p["pose_topic"], self._on_block_pose, 10)
        self.create_subscription(Float32MultiArray, p["score_topic"], self._on_score, 10)
        if self.eef_source == "topic":
            self.create_subscription(PoseStamped, p["eef_pose_topic"], self._on_eef_pose, 10)
            self.tf_buffer = None
        else:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.create_service(Trigger, "~/start", self._on_start)
        self.create_service(Trigger, "~/stop", self._on_stop)
        self.create_service(Trigger, "~/save", self._on_save)
        self.create_service(Trigger, "~/discard", self._on_discard)
        self.create_timer(1.0 / self.cfg.fps, self._tick)
        self.create_timer(1.0, self._heartbeat)
        self._set_state("idle")
        self.get_logger().info(
            f"episode_recorder: {self.cfg.repo_id} at {self.cfg.root}, {self.cfg.fps} Hz, "
            f"{self.cfg.image_size} px crop from {cal_dir}, eef via {self.eef_source}"
        )

    def _declare_parameters(self) -> dict:
        defaults = {
            "mode": "sim",
            "repo_id": "movensys/pusht_cr3a_sim_v1",
            "root": "datasets/movensys/pusht_cr3a_sim_v1",
            "fps": 10,
            "image_size": 256,
            "robot_type": "dobot_cr3a",
            "calibration_dir": "",
            "joint_names": list(rc.JOINT_NAMES),
            "max_image_age": 0.15,
            "tick_tolerance": 0.02,
            "image_topic": "/image_top/rgb",
            "action_topic": "/pusht/target_guarded",
            "joint_topic": "/joint_states",
            "pose_topic": "/pusht/t_block/pose",
            "score_topic": "/pusht/score",
            "eef_source": "tf",
            "eef_pose_topic": "/wmx/moveit2/eef_pose",
            "task.planning_frame": "world_manipulator",
            "task.eef_frame": "Link6",
            "episode_meta": "{}",
        }
        return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

    # ------------------------------------------------------------------ inputs

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_image(self, msg: Image) -> None:
        self.image = msg
        self.image_rx = self._now()

    def _on_action(self, msg: PoseStamped) -> None:
        self.action = (msg.pose.position.x, msg.pose.position.y)
        self.action_rx = time.monotonic()

    def _on_eef_pose(self, msg: PoseStamped) -> None:
        self.eef_pose = msg

    def _on_joints(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position))
        try:
            self.joints = tuple(float(by_name[n]) for n in self.cfg.joint_names)
        except KeyError:
            if not self.joint_warned:
                self.joint_warned = True
                self.get_logger().warning(
                    f"joint_states names {list(msg.name)} lack {self.cfg.joint_names}; "
                    "observation.joints stays empty"
                )

    def _on_block_pose(self, msg: Pose) -> None:
        self.block_pose = pose_to_tuple(msg)

    def _on_score(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            self.coverage, self.iou = float(msg.data[0]), float(msg.data[1])

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

    def _sample(self) -> rc.Sample:
        image = image_t = None
        if self.image is not None:
            image = sc.decode_image(
                self.image.data, self.image.width, self.image.height, self.image.step,
                self.image.encoding,
            )
            stamp = Time.from_msg(self.image.header.stamp).nanoseconds * 1e-9
            image_t = stamp if stamp > 0 else self.image_rx
        return rc.Sample(
            t=self._now(),
            image=image,
            image_t=image_t if image_t is not None else -math.inf,
            eef_xy=self._eef_xy(),
            action_xy=self.action,
            joints=self.joints,
            block_pose=self.block_pose,
            coverage=self.coverage,
            iou=self.iou,
        )

    # ------------------------------------------------------------------ services

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_pub.publish(String(data=state))

    def _heartbeat(self) -> None:
        self.state_pub.publish(String(data=self.state))

    def _on_start(self, _req, res: Trigger.Response) -> Trigger.Response:
        missing = rc.missing_inputs(self._sample())
        if missing:
            res.success = False
            res.message = "missing inputs: " + ", ".join(missing)
            self.get_logger().warning(res.message)
            return res
        if self.buffer is not None and len(self.buffer) and not self.recording:
            self.get_logger().warning("start: dropping an unsaved episode buffer")
        self.buffer = rc.EpisodeBuffer(self.cal, self.cfg)
        self.recording = True
        self._set_state("recording:0")
        res.success = True
        res.message = f"recording episode {self.store.next_episode_index}"
        self.get_logger().info(res.message)
        return res

    def _on_stop(self, _req, res: Trigger.Response) -> Trigger.Response:
        n = self._stop()
        res.success = True
        res.message = f"stopped with {n} frames"
        return res

    def _stop(self) -> int:
        self.recording = False
        n = len(self.buffer) if self.buffer is not None else 0
        self._set_state(f"stopped:{n}")
        return n

    def _on_save(self, _req, res: Trigger.Response) -> Trigger.Response:
        self._stop()
        if self.buffer is None or len(self.buffer) == 0:
            res.success = False
            res.message = "nothing to save"
            return res
        try:
            meta = json.loads(self.get_parameter("episode_meta").value or "{}")
        except json.JSONDecodeError as e:
            meta = {"meta_error": str(e)}
        meta.setdefault("mode", self.get_parameter("mode").value)
        try:
            line = self.store.save(self.buffer, meta)
        except Exception as e:  # noqa: B902  report the failure to the caller, keep the node up
            res.success = False
            res.message = f"save failed: {e}"
            self.get_logger().error(res.message)
            self._set_state("fault:save")
            return res
        self.buffer = None
        self._set_state("idle")
        res.success = True
        res.message = json.dumps(line)
        self.get_logger().info(
            f"saved episode {line['episode_index']}: {line['n_frames']} frames, "
            f"timing_ok={line['timing_ok']} missed={line['missed_ticks']} "
            f"stale={line['stale_frames']} max_image_age={line['max_image_age']}"
        )
        return res

    def _on_discard(self, _req, res: Trigger.Response) -> Trigger.Response:
        n = self._stop()
        self.buffer = None
        self._set_state("idle")
        res.success = True
        res.message = f"discarded {n} frames"
        self.get_logger().info(res.message)
        return res

    # ------------------------------------------------------------------ sampling

    def _tick(self) -> None:
        if not self.recording or self.buffer is None:
            return
        s = self._sample()
        try:
            self.buffer.add(s)
        except ValueError as e:
            self.recording = False
            self._set_state(f"fault:{e}")
            self.get_logger().error(f"recording stopped: {e}")
            return
        self._set_state(f"recording:{len(self.buffer)}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.store.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
