"""
spacemouse_teleop: free 6-DoF arm motion from a 3Dconnexion SpaceMouse through MoveIt Servo.

For parking the rod at push height, the reach checks and general moves; NOT for collection,
where the guard reads the SpaceMouse as its stick (``device:=spacemouse``). Each 50 Hz cycle
feeds ``/spacenav/joy`` to :class:`~movensys_imitation_ros.spacemouse_core.TeleopCore` and
publishes the unitless twist to Servo in TWIST mode.

Single writer: Servo has one owner at a time. This node refuses to switch Servo to TWIST
while anything publishes on Servo's pose topic (the planar guard, ``quest_servo_teleop``) and
holds, with state ``hold:pose_writer``, until that writer is gone; the guard switches Servo
back to POSE when it starts, so stop this node first. Servo's own singularity and collision
stops apply, but the table is not a collision object, so the tool floor here zeroes downward
motion once ``Link6`` is at ``table_z + link6_to_tip_bottom + tip_floor_clearance`` (TF).

Inputs
    /spacenav/joy                     sensor_msgs/Joy          6 axes; buttons: dead-man,
                                                               mode cycle, gripper
    TF planning_frame -> eef_frame                             tool floor
Outputs
    /servo_node/delta_twist_cmds      geometry_msgs/TwistStamped, 50 Hz while engaged
    ~/state                            std_msgs/String          "<state>:<reason>" on change
    ~/mode                             std_msgs/String          motion mode (latched)
Services called
    /servo_node/switch_command_type    TWIST (1), once a writer check passes
    /wmx/set_gripper                   std_srvs/SetBool         gripper toggle button
"""

from __future__ import annotations

import math
import time

from geometry_msgs.msg import TwistStamped
from moveit_msgs.srv import ServoCommandType
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import SetBool
import tf2_ros

from movensys_imitation_ros.guard_core import deadman_held
from movensys_imitation_ros.spacemouse_core import (
    MODES,
    STATE_FAULT,
    TeleopConfig,
    TeleopCore,
    TeleopInputs,
)

SERVO_TWIST = 1


class SpaceMouseTeleop(Node):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__("spacemouse_teleop")
        p = self._declare_parameters()
        self.planning_frame: str = p["task.planning_frame"]
        self.eef_frame: str = p["task.eef_frame"]
        self.frame_id: str = p["frame_id"] or self.planning_frame
        self.z_floor_enabled: bool = p["z_floor_enabled"]
        self.z_floor = p["task.table_z"] + p["task.link6_to_tip_bottom"] + p["tip_floor_clearance"]
        self.require_deadman: bool = p["require_deadman"]
        self.deadman_axis: int = p["deadman_axis"]
        self.deadman_button: int = p["deadman_button"]
        self.deadman_threshold: float = p["deadman_threshold"]
        self.mode_cycle_button: int = p["mode_cycle_button"]
        self.mode_cycle_list: list[str] = list(p["mode_cycle_list"])
        self.gripper_button: int = p["gripper_button"]
        self.pose_topic: str = p["pose_target_topic"]
        self.mode: str = p["motion_mode"]
        if self.mode not in MODES:
            raise ValueError(f"motion_mode must be one of {tuple(MODES)}, got {self.mode!r}")
        for m in self.mode_cycle_list:
            if m not in MODES:
                raise ValueError(f"mode_cycle_list entry {m!r} not in {tuple(MODES)}")

        config = TeleopConfig(
            axis_map=tuple(int(a) for a in p["axis_map"]),
            axis_sign=tuple(float(s) for s in p["axis_sign"]),
            yaw=math.radians(p["yaw_deg"]),
            deadzone=p["deadzone"],
            expo=p["expo"],
            linear_gain=p["linear_gain"],
            angular_gain=p["angular_gain"],
            joy_stale=p["joy_stale"],
            stop_cycles=int(p["stop_cycles"]),
        )
        if len(config.axis_map) != 6 or len(config.axis_sign) != 6:
            raise ValueError("axis_map and axis_sign need six entries")
        if not (0.0 < config.linear_gain <= 1.0 and 0.0 < config.angular_gain <= 1.0):
            raise ValueError("linear_gain and angular_gain must be in (0, 1]")
        self.core = TeleopCore(config)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.joy_axes: list[float] = []
        self.joy_buttons: list[int] = []
        self.joy_rx = -math.inf
        self.twist_mode_set = False
        self.gripper_closed = False
        self.last_state = ""

        self.twist_pub = self.create_publisher(TwistStamped, p["twist_topic"], 10)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, "~/mode", latched)
        self.create_subscription(Joy, p["joy_topic"], self._on_joy, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.switch_client = self.create_client(ServoCommandType, p["servo_switch_service"])
        self.gripper_client = self.create_client(SetBool, p["gripper_service"])
        self.switch_timer = self.create_timer(1.0, self._ensure_twist_mode)
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.period = 1.0 / p["stream_hz"]
        self.create_timer(self.period, self._tick, clock=steady)
        self.mode_pub.publish(String(data=self.mode))
        self.get_logger().info(
            f"spacemouse_teleop: joy {p['joy_topic']} -> {p['twist_topic']} in {self.frame_id}, "
            f"mode {self.mode}, gains linear {config.linear_gain} angular {config.angular_gain} "
            f"(of Servo scale), deadzone {config.deadzone}, floor "
            f"{'z<=%.4f m' % self.z_floor if self.z_floor_enabled else 'OFF'}, "
            f"deadman={self._deadman_desc()}"
        )

    # --- parameters -------------------------------------------------------------------------

    def _declare_parameters(self) -> dict:
        d = TeleopConfig()
        defaults = {
            "task.planning_frame": "world_manipulator",
            "task.eef_frame": "Link6",
            "task.table_z": 0.0431,
            "task.link6_to_tip_bottom": 0.22,
            "joy_topic": "/spacenav/joy",
            "twist_topic": "/servo_node/delta_twist_cmds",
            "servo_switch_service": "/servo_node/switch_command_type",
            "pose_target_topic": "/servo_node/pose_target_cmds",
            "gripper_service": "/wmx/set_gripper",
            "frame_id": "",
            "stream_hz": 50.0,
            "motion_mode": "full",
            "axis_map": list(d.axis_map),
            "axis_sign": [float(v) for v in d.axis_sign],  # DOUBLE_ARRAY, as the yaml
            "yaw_deg": 0.0,
            "deadzone": d.deadzone,
            "expo": d.expo,
            "linear_gain": d.linear_gain,
            "angular_gain": d.angular_gain,
            "joy_stale": d.joy_stale,
            "stop_cycles": d.stop_cycles,
            "require_deadman": True,
            "deadman_axis": -1,
            "deadman_button": 12,
            "deadman_threshold": 0.6,
            "mode_cycle_button": -1,
            "mode_cycle_list": ["full", "translation", "planar", "vertical"],
            "gripper_button": -1,
            "z_floor_enabled": True,
            "tip_floor_clearance": 0.0,
        }
        return {name: self.declare_parameter(name, default).value
                for name, default in defaults.items()}

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        for prm in params:
            if prm.name == "motion_mode":
                if prm.value not in MODES:
                    return SetParametersResult(
                        successful=False, reason=f"motion_mode must be one of {tuple(MODES)}"
                    )
                self._set_mode(prm.value)
            elif prm.name == "require_deadman":
                self.require_deadman = bool(prm.value)
                self.get_logger().warning(f"require_deadman -> {self.require_deadman}")
        return SetParametersResult(successful=True)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.mode_pub.publish(String(data=mode))
        self.get_logger().info(f"motion mode -> {mode}")

    # --- inputs -------------------------------------------------------------------------------

    def _on_joy(self, msg: Joy) -> None:
        buttons = list(msg.buttons)
        if self._rising(buttons, self.mode_cycle_button):
            cycle = self.mode_cycle_list
            idx = cycle.index(self.mode) if self.mode in cycle else -1
            self._set_mode(cycle[(idx + 1) % len(cycle)])
        if self._rising(buttons, self.gripper_button):
            self._toggle_gripper()
        self.joy_axes = list(msg.axes)
        self.joy_buttons = buttons
        self.joy_rx = time.monotonic()

    def _rising(self, buttons: list[int], index: int) -> bool:
        if index < 0 or index >= len(buttons):
            return False
        before = self.joy_buttons[index] if index < len(self.joy_buttons) else 0
        return buttons[index] != 0 and before == 0

    def _toggle_gripper(self) -> None:
        if not self.gripper_client.service_is_ready():
            self.get_logger().warning("gripper service not available")
            return
        req = SetBool.Request()
        req.data = not self.gripper_closed
        self.gripper_closed = req.data

        def done(fut):
            try:
                ok = fut.result().success
            except Exception as exc:  # noqa: B902
                self.get_logger().warning(f"gripper call failed: {exc}")
                return
            self.get_logger().info(
                f"gripper {'closed' if req.data else 'opened'}: {'ok' if ok else 'refused'}"
            )

        self.gripper_client.call_async(req).add_done_callback(done)

    def _deadman_desc(self) -> str:
        if not self.require_deadman:
            return "OFF"
        if self.deadman_button >= 0:
            return f"button {self.deadman_button}"
        return f"axis {self.deadman_axis} >= {self.deadman_threshold}"

    def _deadman(self) -> bool:
        if not self.require_deadman:
            return True
        return deadman_held(
            self.joy_axes, self.joy_buttons, self.deadman_axis, self.deadman_button,
            self.deadman_threshold,
        )

    def _pose_writers(self) -> list[str]:
        return [
            f"{i.node_namespace.rstrip('/')}/{i.node_name}"
            for i in self.get_publishers_info_by_topic(self.pose_topic)
        ]

    def _eef_z(self) -> float | None:
        """Measured end-effector z, or None when TF is missing or stale (0.5 s)."""
        try:
            tf = self.tf_buffer.lookup_transform(self.planning_frame, self.eef_frame, Time())
        except tf2_ros.TransformException:
            return None
        stamp = Time.from_msg(tf.header.stamp)
        if stamp.nanoseconds > 0:
            age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if age > 0.5:
                return None
        return tf.transform.translation.z

    # --- Servo mode -----------------------------------------------------------------------------

    def _ensure_twist_mode(self) -> None:
        if self.twist_mode_set or not self.switch_client.service_is_ready():
            return
        writers = self._pose_writers()
        if writers:
            self.get_logger().warning(
                f"Servo pose topic owned by {writers}; not switching to TWIST",
                throttle_duration_sec=10.0,
            )
            return
        req = ServoCommandType.Request()
        req.command_type = SERVO_TWIST

        def done(fut):
            try:
                ok = fut.result().success
            except Exception as exc:  # noqa: B902
                self.get_logger().warning(f"Servo TWIST switch failed: {exc}")
                return
            if ok:
                self.twist_mode_set = True
                self.get_logger().info("Servo switched to TWIST mode.")
            else:
                self.get_logger().warning("Servo rejected the TWIST switch; retrying.")

        self.switch_client.call_async(req).add_done_callback(done)

    # --- control cycle ------------------------------------------------------------------------

    def _tick(self) -> None:
        if self._pose_writers():
            # The guard or the Quest teleop owns Servo (and has switched it to POSE): hold and
            # re-switch once they are gone.
            self.twist_mode_set = False
            self._publish_state("hold:pose_writer")
            return
        if not self.twist_mode_set:
            self._publish_state("hold:servo_mode")
            return
        eef_z = None
        if self.z_floor_enabled:
            eef_z = self._eef_z()
            if eef_z is None:
                self._publish_state(f"{STATE_FAULT}:eef_stale")
                return
        out = self.core.update(
            TeleopInputs(
                axes=self.joy_axes,
                joy_age=time.monotonic() - self.joy_rx,
                deadman=self._deadman(),
                mode=self.mode,
                eef_z=eef_z,
                z_floor=self.z_floor,
            )
        )
        self._publish_state(f"{out.state}:{out.reason}" if out.reason else out.state)
        if not out.publish:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
         msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z) = out.twist
        self.twist_pub.publish(msg)

    def _publish_state(self, state: str) -> None:
        if state == self.last_state:
            return
        self.last_state = state
        self.state_pub.publish(String(data=state))
        if state.startswith(STATE_FAULT):
            self.get_logger().warning(f"teleop {state}")
        else:
            self.get_logger().info(f"teleop {state}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpaceMouseTeleop()
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
