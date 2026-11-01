"""
episode_manager: the PushT episode loop (plan P2.4, Section 5 "Episode loop").

Per episode: reset (guard to ``pose`` source, lift the tool, home it at the parking point
farthest from the T's start footprint, teleport the T to ``sample_start_pose(seed)``, lower),
record (guard back to the teleop source, recorder ``~/start``), stop on coverage >= 0.95, on
``max_steps`` frames or on a service call, lift, take the authoritative score from the scorer
service, then save or discard. Runs from the venv (``python -m``) and needs the guard, the
scorer and the recorder; Isaac Sim provides the teleport in sim, the operator does it on the
real robot (Phase 4 swaps the reset step).

The blocking :class:`Driver` helpers (lift, go_to, teleport, score) are shared with
``replay_check`` and ``reset_check``; the executor spins in the main thread while the episode
sequence runs in a worker thread.

Services (std_srvs/Trigger)
    ~/start     run the next episode (with ``auto_start`` true, keep running ``n_episodes``)
    ~/stop      end the current episode now, score and save it
    ~/discard   end the current episode now and drop it
    ~/abort     finish or drop the current episode, then stop the run (tool stays lifted)
Outputs
    ~/state     std_msgs/String   idle | reset:<step> | record:<n> | end:<step> | fault:<why>
"""

from __future__ import annotations

import json
import math
import threading
import time

from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.msg import Parameter as ParameterMsg
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
import tf2_ros

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation_ros import manager_core as mc
from movensys_imitation_ros.pusht_scorer import pose_to_tuple
from movensys_imitation_ros.setup_camera import pose_error, pose_msg


DISCOVERY_TIMEOUT = 30.0  # s to wait for Isaac to subscribe to pose_cmd


def spin_in_background(node: Node) -> threading.Thread:
    """Spin ``node`` in a daemon thread so blocking Driver calls can run in the caller."""

    def target():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    thread = threading.Thread(target=target, name="spin", daemon=True)
    thread.start()
    return thread


def stop_spinning(node: Node, thread: threading.Thread) -> None:
    """Shut rclpy down, wait for the spinner and destroy the node.

    A spinner still alive at interpreter exit aborts the process ("terminate called without an
    active exception", exit 134) even after the check itself passed.
    """
    if rclpy.ok():
        rclpy.shutdown()
    thread.join(timeout=5.0)
    node.destroy_node()


class DriverError(RuntimeError):
    """A step of the sequence did not complete (timeout, refused service, missed teleport)."""


class Driver(Node):
    """Blocking helpers over the guard, the scene, the scorer and the recorder."""

    def __init__(self, name: str = "episode_manager") -> None:
        super().__init__(name)
        p = self._declare_driver_parameters()
        self.planning_frame: str = p["task.planning_frame"]
        self.eef_frame: str = p["task.eef_frame"]
        self.block_pose: tuple[float, float, float] | None = None
        self.block_pose_t = -math.inf
        self.coverage = math.nan
        self.lift_state = ""
        self.recorder_state = ""
        self.recorder_frames = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Pose, p["pose_topic"], self._on_block_pose, 10)
        self.create_subscription(Float32MultiArray, p["score_topic"], self._on_score, 10)
        self.create_subscription(String, "/planar_guard/lift_state", self._on_lift, latched)
        self.create_subscription(String, "/episode_recorder/state", self._on_recorder, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_pub = self.create_publisher(PoseStamped, "/pusht/target_raw", 10)
        self.path_pub = self.create_publisher(Path, "/pusht/waypoints", 10)
        self.teleport_pub = self.create_publisher(Pose, "/pusht/t_block/pose_cmd", 10)
        self.lift_client = self.create_client(SetBool, "/planar_guard/lift")
        self.score_client = self.create_client(Trigger, "/pusht_scorer/score")
        self.recorder_clients = {
            cmd: self.create_client(Trigger, f"/episode_recorder/{cmd}")
            for cmd in ("start", "stop", "save", "discard")
        }
        self.guard_params = AsyncParameterClient(self, "planar_guard")
        self._deadman_before: bool | None = None
        self.recorder_params = AsyncParameterClient(self, "episode_recorder")

    def _declare_driver_parameters(self) -> dict:
        defaults = {
            "task.planning_frame": "world_manipulator",
            "task.eef_frame": "Link6",
            "pose_topic": "/pusht/t_block/pose",
            "score_topic": "/pusht/score",
            "service_timeout": 5.0,
            "scripted_deadman": False,
            "lift_timeout": 5.0,
            "home_tolerance": 0.008,
            "home_timeout": 20.0,
            "teleport_settle": 1.5,
        }
        return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

    # ------------------------------------------------------------------ callbacks

    def _on_block_pose(self, msg: Pose) -> None:
        self.block_pose = pose_to_tuple(msg)
        self.block_pose_t = time.monotonic()

    def _on_score(self, msg: Float32MultiArray) -> None:
        if msg.data:
            self.coverage = float(msg.data[0])

    def _on_lift(self, msg: String) -> None:
        self.lift_state = msg.data

    def _on_recorder(self, msg: String) -> None:
        self.recorder_state = msg.data
        if ":" in msg.data and msg.data.split(":")[0] in ("recording", "stopped"):
            try:
                self.recorder_frames = int(msg.data.split(":")[1])
            except ValueError:
                pass

    # ------------------------------------------------------------------ primitives

    def wait_until(self, pred, timeout: float, what: str, period: float = 0.02) -> None:
        t0 = time.monotonic()
        while not pred():
            if not rclpy.ok():
                raise DriverError("shutdown")
            if time.monotonic() - t0 > timeout:
                raise DriverError(f"timeout after {timeout:.1f} s waiting for {what}")
            time.sleep(period)

    def call(self, client, request, timeout: float | None = None):
        timeout = timeout or self.get_parameter("service_timeout").value
        if not client.wait_for_service(timeout_sec=timeout):
            raise DriverError(f"service {client.srv_name} not available")
        future = client.call_async(request)
        self.wait_until(future.done, timeout, client.srv_name)
        return future.result()

    def set_param(self, client: AsyncParameterClient, name: str, value) -> None:
        timeout = self.get_parameter("service_timeout").value
        if not client.wait_for_services(timeout_sec=timeout):
            raise DriverError(f"parameter services of {client.remote_node_name} not available")
        future = client.set_parameters([Parameter(name, value=value)])
        self.wait_until(future.done, timeout, f"set {name} on {client.remote_node_name}")
        result = future.result().results[0]
        if not result.successful:
            raise DriverError(
                f"{client.remote_node_name} refused {name}={value!r}: {result.reason}"
            )

    def get_param(self, client: AsyncParameterClient, name: str):
        timeout = self.get_parameter("service_timeout").value
        if not client.wait_for_services(timeout_sec=timeout):
            raise DriverError(f"parameter services of {client.remote_node_name} not available")
        future = client.get_parameters([name])
        self.wait_until(future.done, timeout, f"get {name} from {client.remote_node_name}")
        return Parameter.from_parameter_msg(
            ParameterMsg(name=name, value=future.result().values[0])
        ).value

    def scripted(self, active: bool) -> None:
        """Drop the guard's dead-man for the manager's own moves, restore it for teleop.

        The grip is a teleop safety; the lift, home and lower moves are scripted, and with
        nobody holding the grip they never happened (go_to timed out 208 mm away, 2026-10-28).
        ``scripted_deadman: true`` keeps the grip required throughout (real robot, if wanted).
        """
        if self.get_parameter("scripted_deadman").value:
            return
        if active:
            if self._deadman_before is None:
                self._deadman_before = bool(self.get_param(self.guard_params, "require_deadman"))
            if self._deadman_before:
                self.set_param(self.guard_params, "require_deadman", False)
        elif self._deadman_before is not None:
            self.set_param(self.guard_params, "require_deadman", self._deadman_before)
            self._deadman_before = None

    def eef_xy(self) -> tuple[float, float] | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.planning_frame, self.eef_frame, Time())
        except tf2_ros.TransformException:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    # ------------------------------------------------------------------ guard

    def set_guard_source(self, source: str) -> None:
        self.set_param(self.guard_params, "source", source)

    def lift(self, up: bool) -> None:
        res = self.call(self.lift_client, SetBool.Request(data=up))
        if not res.success:
            raise DriverError(f"lift refused: {res.message}")
        want = "up" if up else "down"
        self.wait_until(
            lambda: self.lift_state == want, self.get_parameter("lift_timeout").value,
            f"lift_state {want} (now {self.lift_state!r})",
        )

    def publish_target(self, xy) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.planning_frame
        msg.pose.position.x, msg.pose.position.y = float(xy[0]), float(xy[1])
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)

    def go_to(self, xy, tolerance: float | None = None, timeout: float | None = None) -> float:
        """Stream ``xy`` to the guard (``pose`` source) until the end effector is there."""
        tolerance = tolerance or self.get_parameter("home_tolerance").value
        timeout = timeout or self.get_parameter("home_timeout").value
        t0 = time.monotonic()
        while True:
            self.publish_target(xy)
            eef = self.eef_xy()
            if eef is not None:
                d = math.hypot(eef[0] - xy[0], eef[1] - xy[1])
                if d <= tolerance:
                    return d
            if not rclpy.ok():
                raise DriverError("shutdown")
            if time.monotonic() - t0 > timeout:
                where = "no TF" if eef is None else f"{d * 1000:.1f} mm away"
                raise DriverError(f"go_to {tuple(round(v, 3) for v in xy)} timed out, {where}")
            time.sleep(0.05)

    def publish_path(self, waypoints) -> None:
        """``(t, x, y)`` waypoints (seconds on the node clock) as a timed Path to the guard."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.planning_frame
        for t, x, y in waypoints:
            ps = PoseStamped()
            ps.header.frame_id = self.planning_frame
            ps.header.stamp = Time(seconds=t).to_msg()
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ scene

    def teleport(self, pose, settle: float | None = None, retries: int = 3) -> tuple[float, float]:
        """Teleport the T and verify the published pose (2 mm, 1 deg); returns the error."""
        settle = settle or self.get_parameter("teleport_settle").value
        # Isaac (fastrtps) discovers a fresh cyclonedds publisher in 5 to 10 s on the lab box
        # (first reset_check teleport failed at a 5 s cap, 2026-10-28), so allow 30 s once.
        t0 = time.monotonic()
        while self.teleport_pub.get_subscription_count() == 0:
            waited = time.monotonic() - t0
            if waited > DISCOVERY_TIMEOUT:
                raise DriverError(
                    f"no subscriber on /pusht/t_block/pose_cmd after {waited:.0f} s; "
                    "is the scene playing?"
                )
            if waited > 3.0 and int(waited * 10) % 50 == 0:
                self.get_logger().info(f"waiting for the pose_cmd subscriber ({waited:.0f} s)")
            time.sleep(0.1)
        msg = pose_msg(pose)
        for attempt in range(retries):
            for _ in range(5):
                self.teleport_pub.publish(msg)
                time.sleep(0.1)
            time.sleep(settle)
            if self.block_pose is None:
                raise DriverError("no block pose on the pose topic")
            d, dyaw = pose_error(pose_msg(self.block_pose), pose)
            if d < 0.002 and dyaw < math.radians(1):
                return d, dyaw
            self.get_logger().warning(
                f"teleport attempt {attempt + 1} missed by {d * 1000:.1f} mm, "
                f"{math.degrees(dyaw):.1f} deg"
            )
        raise DriverError(f"teleport not reached after {retries} tries ({d * 1000:.1f} mm off)")

    def block_motion(self, window: float) -> tuple[float, float]:
        """How far the T moves over ``window`` seconds (m, rad)."""
        if self.block_pose is None:
            raise DriverError("no block pose")
        a = self.block_pose
        time.sleep(window)
        return mc.pose_difference(self.block_pose, a)

    # ------------------------------------------------------------------ scorer, recorder

    def score(self) -> dict:
        res = self.call(self.score_client, Trigger.Request())
        try:
            out = json.loads(res.message)
        except json.JSONDecodeError as e:
            raise DriverError(f"scorer returned no JSON: {e}") from e
        out["service_success"] = bool(res.success)
        return out

    def recorder(self, cmd: str) -> str:
        res = self.call(self.recorder_clients[cmd], Trigger.Request(), timeout=60.0)
        if not res.success:
            raise DriverError(f"recorder {cmd} failed: {res.message}")
        return res.message

    def set_episode_meta(self, meta: dict) -> None:
        self.set_param(self.recorder_params, "episode_meta", json.dumps(meta))


class EpisodeManager(Driver):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__("episode_manager")
        p = self._declare_parameters()
        self.cfg = mc.ManagerConfig(
            success_threshold=p["success_threshold"],
            max_steps=int(p["max_steps"]),
            min_home_clearance=p["min_home_clearance"],
            home_candidates=mc.pairs(p["task.home_candidates"]),
            discard_on_timing_fault=p["discard_on_timing_fault"],
            keep_failures=p["keep_failures"],
        )
        self.sheet_origin = tuple(float(v) for v in p["task.area_center"])
        self.operator: str = p["operator"]
        self.mode: str = p["mode"]
        self.teleop_source: str = p["teleop_source"]
        self.input_device: str = p["input_device"]
        self.seed = int(p["seed_start"])
        self.n_episodes = int(p["n_episodes"])
        self.auto_start: bool = p["auto_start"]
        self.lower_settle: float = p["lower_settle"]
        self.fps: float = p["task.rates.policy_hz"]

        self.start_event = threading.Event()
        self.stop_requested = False
        self.discard_requested = False
        self.abort_requested = False
        self.state = ""
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.create_service(Trigger, "~/start", self._on_start)
        self.create_service(Trigger, "~/stop", self._on_stop)
        self.create_service(Trigger, "~/discard", self._on_discard)
        self.create_service(Trigger, "~/abort", self._on_abort)
        self.create_timer(1.0, lambda: self.state_pub.publish(String(data=self.state)))
        self._set_state("idle")
        self.get_logger().info(
            f"episode_manager: seeds from {self.seed}, {self.n_episodes or 'unlimited'} episodes, "
            f"teleop source {self.teleop_source!r}, operator {self.operator!r}, "
            f"auto_start={self.auto_start}; call ~/start"
        )
        self.worker = threading.Thread(target=self.run, name="episode_loop", daemon=True)

    def _declare_parameters(self) -> dict:
        defaults = {
            "mode": "sim",
            "operator": "",
            "teleop_source": "stick",
            "input_device": "",  # sidecar provenance: quest | spacemouse (collect.launch.py)
            "seed_start": 1000,
            "n_episodes": 10,
            "auto_start": False,
            "success_threshold": mc.ManagerConfig.success_threshold,
            "max_steps": 300,
            "min_home_clearance": 0.04,
            "discard_on_timing_fault": True,
            "keep_failures": True,
            "lower_settle": 0.5,
            "task.home_candidates": [v for xy in mc.HOME_CANDIDATES for v in xy],
            "task.area_center": [-0.03, -0.05],
            "task.rates.policy_hz": 10.0,
        }
        return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

    # ------------------------------------------------------------------ services

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_pub.publish(String(data=state))
            self.get_logger().info(f"state {state}")

    def _on_start(self, _req, res: Trigger.Response) -> Trigger.Response:
        self.abort_requested = False
        self.start_event.set()
        res.success = True
        res.message = f"next seed {self.seed}"
        return res

    def _on_stop(self, _req, res: Trigger.Response) -> Trigger.Response:
        self.stop_requested = True
        res.success = True
        res.message = "stop requested"
        return res

    def _on_discard(self, _req, res: Trigger.Response) -> Trigger.Response:
        self.discard_requested = True
        res.success = True
        res.message = "discard requested"
        return res

    def _on_abort(self, _req, res: Trigger.Response) -> Trigger.Response:
        self.abort_requested = True
        self.stop_requested = True
        self.start_event.set()
        res.success = True
        res.message = "abort requested"
        return res

    # ------------------------------------------------------------------ the loop

    def run(self) -> None:
        done = 0
        while rclpy.ok():
            self._set_state("idle")
            self.start_event.wait()
            self.start_event.clear()
            if self.abort_requested or not rclpy.ok():
                continue
            try:
                self.episode(self.seed)
                self.scripted(False)
            except DriverError as e:
                self._set_state(f"fault:{e}")
                self.get_logger().error(f"episode {self.seed} aborted: {e}")
                self._safe_stop()
                continue
            done += 1
            self.seed += 1
            if self.auto_start and not self.abort_requested and (
                self.n_episodes == 0 or done < self.n_episodes
            ):
                self.start_event.set()

    def _safe_stop(self) -> None:
        for step in (lambda: self.recorder("discard"), lambda: self.set_guard_source("pose"),
                     lambda: self.scripted(True), lambda: self.lift(True),
                     lambda: self.scripted(False)):
            try:
                step()
            except DriverError as e:
                self.get_logger().warning(f"safe stop: {e}")

    def episode(self, seed: int) -> None:
        self.stop_requested = self.discard_requested = False
        cfg = self.cfg
        start_pose = g.sample_start_pose(seed)
        home, clearance = mc.choose_home(
            cfg.home_candidates, start_pose, cfg.min_home_clearance, self.sheet_origin
        )
        self.get_logger().info(
            f"episode seed {seed}: T start {tuple(round(v, 3) for v in start_pose)}, "
            f"home {home} (clearance {clearance * 1000:.0f} mm)"
        )

        # Reset: hold the arm, lift, park, teleport the T, lower.
        self._set_state("reset:lift")
        self.set_guard_source("pose")
        self.scripted(True)
        self.lift(True)
        self._set_state("reset:home")
        self.go_to(home)
        self._set_state("reset:teleport")
        d, dyaw = self.teleport(start_pose)
        moved = self.block_motion(0.5)
        if moved[0] > 0.0005 or moved[1] > math.radians(0.2):
            raise DriverError(f"T still moving after teleport ({moved[0] * 1000:.2f} mm in 0.5 s)")
        self._set_state("reset:lower")
        self.lift(False)
        time.sleep(self.lower_settle)

        # Record.
        meta = mc.episode_meta(
            seed, start_pose, home, self.operator, self.mode,
            {
                "teleport_error_mm": round(d * 1000, 2),
                "teleport_error_deg": round(math.degrees(dyaw), 2),
                "input_device": self.input_device,
            },
        )
        self.set_episode_meta(meta)
        self.scripted(False)
        self.set_guard_source(self.teleop_source)
        self.recorder("start")
        self.recorder_frames = 0
        t_start = time.monotonic()
        reason = None
        while reason is None:
            self._set_state(f"record:{self.recorder_frames}")
            reason = mc.stop_reason(
                self.coverage, self.recorder_frames, cfg, self.stop_requested,
                self.discard_requested,
            )
            if self.recorder_state.startswith("fault"):
                raise DriverError(f"recorder {self.recorder_state}")
            if reason is None:
                time.sleep(0.02)
        wall = time.monotonic() - t_start
        self.recorder("stop")

        # End: hold, lift, authoritative score, keep or drop.
        self._set_state("end:score")
        self.set_guard_source("pose")
        self.scripted(True)
        self.lift(True)
        time.sleep(0.5)
        score = self.score()
        timing_ok = True
        outcome = mc.decide_outcome(reason, score, timing_ok, cfg)
        end_meta = dict(meta)
        end_meta.update(
            stop_reason=reason, success=outcome.success, end_score=score,
            final_coverage=outcome.coverage, final_iou=outcome.iou,
            wall_seconds=round(wall, 2), end_block_pose=self.block_pose,
        )
        self.set_episode_meta(end_meta)
        if not outcome.save:
            self._set_state("end:discard")
            self.recorder("discard")
            self.get_logger().info(f"episode seed {seed} discarded: {outcome.reason}")
            return
        self._set_state("end:save")
        line = json.loads(self.recorder("save"))
        if not line.get("timing_ok", True) and cfg.discard_on_timing_fault:
            # The recorder's verdict arrives with the save; the episode is on disk, so flag it
            # loudly in the sidecar for P2.6 to drop rather than rewriting the dataset here.
            self.get_logger().warning(
                f"episode {line['episode_index']} has a timing fault (missed "
                f"{line['missed_ticks']}, stale {line['stale_frames']}); marked in the sidecar"
            )
        self.get_logger().info(
            f"episode {line['episode_index']} saved: seed {seed}, {line['n_frames']} frames, "
            f"{reason}, coverage {outcome.coverage:.3f}, success {outcome.success}, "
            f"time to success {line.get('time_to_success')}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeManager()
    node.worker.start()
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
