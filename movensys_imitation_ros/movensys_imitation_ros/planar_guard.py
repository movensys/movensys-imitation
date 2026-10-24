"""
planar_guard: the one safety and consistency layer between teleop or policy and MoveIt Servo.

Plan P2.2. Runs on the host venv with standard messages only. Each 50 Hz cycle it feeds the
measured end-effector pose, the Quest Joy state and the current pose command to
:class:`~movensys_imitation_ros.guard_core.GuardCore`, then publishes the guarded target with the
latched z and orientation to ``/servo_node/pose_target_cmds`` and ``/pusht/target_guarded``.

Inputs
    /quest_pose_publisher/joy_right   sensor_msgs/Joy          thumbstick and grip (dead-man)
    /pusht/target_raw                 geometry_msgs/PoseStamped  absolute target (source "pose")
    /pusht/waypoints                  nav_msgs/Path            timed waypoints, interpolated
    TF planning_frame -> eef_frame, or /wmx/moveit2/eef_pose   measured end effector
    /servo_node/status                moveit_msgs/ServoStatus  halt detection
Outputs
    /servo_node/pose_target_cmds, /pusht/target_guarded   geometry_msgs/PoseStamped, 50 Hz
    ~/state                            std_msgs/String          "<state>:<reason>" on change
Services
    ~/relatch                          std_srvs/Trigger         re-latch z and orientation
    /servo_node/switch_command_type    called once with POSE (2)
"""

from __future__ import annotations

import math
import time

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
from nav_msgs.msg import Path
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros

from movensys_imitation_ros.guard_core import (
    GuardConfig,
    GuardCore,
    Inputs,
    SOURCES,
    STATE_FAULT,
)

SERVO_POSE = 2
SERVO_HALT_CODES = (
    ServoStatus.HALT_FOR_SINGULARITY,
    ServoStatus.HALT_FOR_COLLISION,
    ServoStatus.JOINT_BOUND,
)


class PlanarGuard(Node):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__("planar_guard")
        p = self._declare_parameters()
        self.planning_frame: str = p["task.planning_frame"]
        self.eef_frame: str = p["task.eef_frame"]
        self.table_z: float = p["task.table_z"]
        self.link6_to_tip: float = p["task.link6_to_tip_bottom"]
        self.tip_clearance_nominal: float = p["task.tip_clearance_nominal"]
        self.z_setpoint: float = p["z_setpoint"]
        self.stick_axes = [int(a) for a in p["stick_axes"]]
        self.require_deadman: bool = p["require_deadman"]
        self.deadman_axis: int = p["deadman_axis"]
        self.deadman_threshold: float = p["deadman_threshold"]

        config = GuardConfig(
            area_center=tuple(p["task.area_center"]),
            area_size=tuple(p["task.area_size"]),
            step_limit=p["step_limit"],
            eef_distance_limit=p["eef_distance_limit"],
            max_speed=p["max_speed"],
            deadzone=p["deadzone"],
            expo=p["expo"],
            stick_yaw=math.radians(p["stick_yaw_deg"]),
            stick_invert_x=p["stick_invert_x"],
            stick_invert_y=p["stick_invert_y"],
            joy_stale=p["joy_stale"],
            eef_stale=p["eef_stale"],
            pose_stale=p["pose_stale"],
        )
        source = p["source"]
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        self.core = GuardCore(config, source)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Latest inputs. Ages are measured on the monotonic clock at arrival, except TF, whose
        # stamps and the node clock share /clock.
        self.eef_pose: PoseStamped | None = None
        self.eef_rx = 0.0
        self.joy_axes: list[float] = []
        self.joy_rx = -math.inf
        self.pose_cmd: tuple[float, float] | None = None
        self.pose_rx = -math.inf
        self.path: list[tuple[float, float, float]] = []  # (t_sec, x, y)
        self.servo_halted = False
        self.servo_code = ServoStatus.NO_WARNING
        self.latched_z: float | None = None
        self.latched_q: tuple[float, float, float, float] | None = None  # x, y, z, w
        self.last_state = ""
        self.pose_mode_set = False

        # ROS entities.
        self.servo_pub = self.create_publisher(PoseStamped, p["servo_pose_topic"], 10)
        self.guarded_pub = self.create_publisher(PoseStamped, "/pusht/target_guarded", 10)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.create_subscription(Joy, p["joy_topic"], self._on_joy, 10)
        self.create_subscription(PoseStamped, "/pusht/target_raw", self._on_target_raw, 10)
        self.create_subscription(Path, "/pusht/waypoints", self._on_waypoints, 10)
        self.create_subscription(ServoStatus, p["servo_status_topic"], self._on_servo_status, 10)
        self.eef_source: str = p["eef_source"]
        if self.eef_source == "topic":
            self.create_subscription(PoseStamped, p["eef_pose_topic"], self._on_eef_pose, 10)
            self.tf_buffer = None
        else:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_service(Trigger, "~/relatch", self._on_relatch)
        self.switch_client = self.create_client(ServoCommandType, p["servo_switch_service"])
        self.switch_timer = self.create_timer(1.0, self._ensure_pose_mode)

        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.period = 1.0 / p["stream_hz"]
        self.create_timer(self.period, self._tick, clock=steady)
        self.get_logger().info(
            f"planar_guard: source={source} eef={self.eef_source} area centre "
            f"{config.area_center} size {config.area_size} step {config.step_limit} m "
            f"lead {config.eef_distance_limit} m max_speed {config.max_speed} m/s "
            f"deadman={'on' if self.require_deadman else 'OFF'}"
        )

    # --- parameters -------------------------------------------------------------------------

    def _declare_parameters(self) -> dict:
        d = GuardConfig()
        defaults = {
            "task.planning_frame": "world_manipulator",
            "task.eef_frame": "Link6",
            "task.area_center": list(d.area_center),
            "task.area_size": list(d.area_size),
            "task.table_z": 0.0431,
            "task.link6_to_tip_bottom": 0.22,
            "task.tip_clearance_nominal": 0.010,
            "source": "stick",
            "z_setpoint": -1.0,
            "eef_source": "tf",
            "eef_pose_topic": "/wmx/moveit2/eef_pose",
            "joy_topic": "/quest_pose_publisher/joy_right",
            "stick_axes": [0, 1],
            "stick_yaw_deg": 0.0,
            "stick_invert_x": False,
            "stick_invert_y": False,
            "deadzone": d.deadzone,
            "expo": d.expo,
            "max_speed": d.max_speed,
            "require_deadman": True,
            "deadman_axis": 3,
            "deadman_threshold": 0.6,
            "step_limit": d.step_limit,
            "eef_distance_limit": d.eef_distance_limit,
            "joy_stale": d.joy_stale,
            "eef_stale": d.eef_stale,
            "pose_stale": d.pose_stale,
            "servo_pose_topic": "/servo_node/pose_target_cmds",
            "servo_switch_service": "/servo_node/switch_command_type",
            "servo_status_topic": "/servo_node/status",
            "stream_hz": 50.0,
        }
        out = {}
        for name, default in defaults.items():
            out[name] = self.declare_parameter(name, default).value
        return out

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        for prm in params:
            if prm.name == "source":
                if prm.value not in SOURCES:
                    return SetParametersResult(
                        successful=False, reason=f"source must be one of {SOURCES}"
                    )
                self.core.set_source(prm.value)
                self.get_logger().info(f"source -> {prm.value}")
        return SetParametersResult(successful=True)

    # --- subscriptions ----------------------------------------------------------------------

    def _on_joy(self, msg: Joy) -> None:
        self.joy_axes = list(msg.axes)
        self.joy_rx = time.monotonic()

    def _on_eef_pose(self, msg: PoseStamped) -> None:
        if not self._frame_ok(msg.header.frame_id, "eef_pose"):
            return
        self.eef_pose = msg
        self.eef_rx = time.monotonic()

    def _on_target_raw(self, msg: PoseStamped) -> None:
        if not self._frame_ok(msg.header.frame_id, "target_raw"):
            return
        self.path = []
        self.pose_cmd = (msg.pose.position.x, msg.pose.position.y)
        self.pose_rx = time.monotonic()

    def _on_waypoints(self, msg: Path) -> None:
        if not self._frame_ok(msg.header.frame_id, "waypoints"):
            return
        self.path = [
            (Time.from_msg(ps.header.stamp).nanoseconds * 1e-9, ps.pose.position.x,
             ps.pose.position.y)
            for ps in msg.poses
        ]
        self.path.sort(key=lambda w: w[0])

    def _on_servo_status(self, msg: ServoStatus) -> None:
        halted = msg.code in SERVO_HALT_CODES
        if msg.code != self.servo_code:
            # rclpy pins the severity per call site, so two sites.
            if halted:
                self.get_logger().warning(f"servo status {msg.code}: {msg.message}")
            else:
                self.get_logger().info(f"servo status {msg.code}: {msg.message}")
        self.servo_code = msg.code
        self.servo_halted = halted

    def _frame_ok(self, frame_id: str, what: str) -> bool:
        if frame_id in ("", self.planning_frame):
            return True
        self.get_logger().warning(
            f"{what} in frame {frame_id!r}, expected {self.planning_frame!r}; ignored",
            throttle_duration_sec=5.0,
        )
        return False

    # --- services -----------------------------------------------------------------------------

    def _on_relatch(self, _req, res: Trigger.Response) -> Trigger.Response:
        self.latched_z = None
        self.latched_q = None
        self.core.reset()
        res.success = True
        res.message = "plane will re-latch on the next end-effector pose"
        self.get_logger().info(res.message)
        return res

    def _ensure_pose_mode(self) -> None:
        if self.pose_mode_set:
            self.switch_timer.cancel()
            return
        if not self.switch_client.service_is_ready():
            return
        req = ServoCommandType.Request()
        req.command_type = SERVO_POSE
        future = self.switch_client.call_async(req)

        def done(fut):
            try:
                ok = fut.result().success
            except Exception as exc:  # noqa: B902  (service call failed; retry next tick)
                self.get_logger().warning(f"Servo POSE switch failed: {exc}")
                return
            if ok:
                self.pose_mode_set = True
                self.get_logger().info("Servo switched to POSE mode.")
            else:
                self.get_logger().warning("Servo rejected the POSE switch; retrying.")

        future.add_done_callback(done)

    # --- control cycle ------------------------------------------------------------------------

    def _measured_eef(self) -> tuple[tuple[float, float] | None, float]:
        """Return the measured EEF (x, y) and its age in seconds."""
        if self.eef_source == "topic":
            if self.eef_pose is None:
                return None, math.inf
            pos = self.eef_pose.pose.position
            return (pos.x, pos.y), time.monotonic() - self.eef_rx
        try:
            tf = self.tf_buffer.lookup_transform(self.planning_frame, self.eef_frame, Time())
        except tf2_ros.TransformException:
            return None, math.inf
        ps = PoseStamped()
        ps.header = tf.header
        t, r = tf.transform.translation, tf.transform.rotation
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = t.x, t.y, t.z
        ps.pose.orientation.x, ps.pose.orientation.y = r.x, r.y
        ps.pose.orientation.z, ps.pose.orientation.w = r.z, r.w
        self.eef_pose = ps
        stamp = Time.from_msg(tf.header.stamp)
        age = 0.0
        if stamp.nanoseconds > 0:
            age = max(0.0, (self.get_clock().now() - stamp).nanoseconds * 1e-9)
        return (t.x, t.y), age

    def _latch(self) -> None:
        pose = self.eef_pose.pose
        z = pose.position.z if self.z_setpoint <= 0.0 else self.z_setpoint
        self.latched_z = z
        q = pose.orientation
        self.latched_q = (q.x, q.y, q.z, q.w)
        clearance = z - self.link6_to_tip - self.table_z
        msg = (
            f"plane latched: z={z:.4f} m ({'measured' if self.z_setpoint <= 0 else 'setpoint'}), "
            f"tip clearance {clearance * 1000:.1f} mm above the table (nominal "
            f"{self.tip_clearance_nominal * 1000:.0f}), orientation xyzw=({q.x:.3f}, {q.y:.3f}, "
            f"{q.z:.3f}, {q.w:.3f})"
        )
        if abs(clearance - self.tip_clearance_nominal) > 0.005:
            self.get_logger().warning(msg + " -- outside +-5 mm of nominal; check the height")
        else:
            self.get_logger().info(msg)

    def _deadman(self) -> bool:
        if not self.require_deadman:
            return True
        if self.deadman_axis >= len(self.joy_axes):
            return False
        return self.joy_axes[self.deadman_axis] >= self.deadman_threshold

    def _stick(self) -> tuple[float, float] | None:
        ax, ay = self.stick_axes
        if max(ax, ay) >= len(self.joy_axes):
            return None
        return (self.joy_axes[ax], self.joy_axes[ay])

    def _pose_command(self) -> tuple[tuple[float, float] | None, float]:
        """Return the current absolute command and its age: path sample or last raw target."""
        if self.path:
            now = self.get_clock().now().nanoseconds * 1e-9
            first, last = self.path[0], self.path[-1]
            if now <= first[0]:
                return (first[1], first[2]), 0.0
            if now >= last[0]:
                return (last[1], last[2]), now - last[0]
            for (t0, x0, y0), (t1, x1, y1) in zip(self.path, self.path[1:]):
                if t0 <= now <= t1:
                    a = 0.0 if t1 == t0 else (now - t0) / (t1 - t0)
                    return (x0 + a * (x1 - x0), y0 + a * (y1 - y0)), 0.0
        if self.pose_cmd is None:
            return None, math.inf
        return self.pose_cmd, time.monotonic() - self.pose_rx

    def _tick(self) -> None:
        eef_xy, eef_age = self._measured_eef()
        if eef_xy is not None and self.latched_z is None and eef_age <= self.core.config.eef_stale:
            self._latch()
        pose_cmd, pose_age = self._pose_command()
        out = self.core.update(
            Inputs(
                dt=self.period,
                eef_xy=eef_xy,
                eef_age=eef_age,
                servo_halted=self.servo_halted,
                deadman=self._deadman(),
                joy_age=time.monotonic() - self.joy_rx,
                stick=self._stick(),
                pose_cmd=pose_cmd,
                pose_age=pose_age,
            )
        )
        state = f"{out.state}:{out.reason}"
        if state != self.last_state:
            self.last_state = state
            self.state_pub.publish(String(data=state))
            if out.state == STATE_FAULT:
                self.get_logger().warning(f"guard {state}")
            else:
                self.get_logger().info(f"guard {state}")
        if not out.publish or out.target is None or self.latched_z is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.planning_frame
        msg.pose.position.x, msg.pose.position.y = out.target
        msg.pose.position.z = self.latched_z
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = self.latched_q
        self.servo_pub.publish(msg)
        self.guarded_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanarGuard()
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
