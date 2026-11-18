#!/usr/bin/env python3
"""Camera alignment helper: live RealSense color view with persistent pixel marks.

Use it to put the camera back exactly where it was if it gets bumped during data
collection: mark a few distinctive scene points once, and later move the camera
until those points sit under their marks again.

Controls
  hover         show pixel coordinates (and the color under the cursor)
  left click    add a mark at the cursor
  right click   remove the mark nearest the cursor
  r             save the current frame as the reference image
  o             toggle a 50% overlay of the reference image (ghost alignment)
  m             hide/show the marks (to see the scene underneath)
  c             clear all marks (press twice to confirm)
  q / Esc       quit

Marks and the reference image are saved on every change, so they are still there
the next time the program starts. reference_marked.png (the reference with the
marks drawn on it) is rewritten whenever either one changes.

Sources
  default       open the D455 color node directly over V4L2 (no SDK needed).
                The device can only be opened by one process, so this does not
                work while the realsense2_camera ROS node is running.
  --ros-topic   subscribe to the ROS image topic instead; use this while data
                collection is running (source env.bash first).
"""

import argparse
import glob
import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

# The D455's color sensor (OV9782) tops out at 1280x800; it cannot stream 1080p.
WIDTH, HEIGHT = 1280, 800
DEFAULT_DIR = Path(__file__).resolve().parent  # marks.json + reference.png live next to this script
WINDOW = "camera alignment"


# ---------------------------------------------------------------- sources ----
def find_realsense_color_node():
    """Return the /dev/video index whose stream gives full 1280x800 color."""
    for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        if "RealSense" not in Path(name_file).read_text():
            continue
        idx = int(Path(name_file).parent.name.replace("video", ""))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        ok, frame = cap.read()
        cap.release()
        if ok and frame.shape[:2] == (HEIGHT, WIDTH):
            return idx
    return None


class V4L2Source:
    def __init__(self, device):
        if device is None:
            device = find_realsense_color_node()
            if device is None:
                raise SystemExit(
                    "No RealSense color node found (is the ROS camera node holding it? "
                    "then use --ros-topic)."
                )
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open /dev/video{device}")
        print(f"Opened /dev/video{device}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class RosSource:
    def __init__(self, topic):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node("camera_alignment")
        self._frame = None
        self._lock = threading.Lock()
        self.node.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)
        self._thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._thread.start()
        print(f"Subscribed to {topic}")

    def _cb(self, msg):
        buf = np.frombuffer(msg.data, np.uint8)
        enc = msg.encoding.lower()
        if enc in ("rgb8", "bgr8"):
            img = buf.reshape(msg.height, msg.step)[:, : msg.width * 3]
            img = img.reshape(msg.height, msg.width, 3)
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc in ("yuv422_yuy2", "yuyv"):
            img = cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
        else:
            self.node.get_logger().warn(f"Unsupported encoding {msg.encoding}", once=True)
            return
        with self._lock:
            self._frame = img.copy()

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self.node.destroy_node()
        self._rclpy.try_shutdown()


# ---------------------------------------------------------------- storage ----
class Store:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.marks_file = self.dir / "marks.json"
        self.ref_file = self.dir / "reference.png"
        self.marked_file = self.dir / "reference_marked.png"
        self.marks = []
        if self.marks_file.exists():
            self.marks = [tuple(m) for m in json.loads(self.marks_file.read_text())["marks"]]
        self.reference = cv2.imread(str(self.ref_file)) if self.ref_file.exists() else None

    def save_marks(self):
        tmp = self.marks_file.with_suffix(".tmp")
        data = {"width": WIDTH, "height": HEIGHT, "marks": self.marks}
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.marks_file)  # atomic: a crash never leaves a half-written file
        self.save_marked()

    def save_reference(self, frame):
        cv2.imwrite(str(self.ref_file), frame)
        self.reference = frame.copy()
        self.save_marked()

    def save_marked(self):
        """Write the reference image with the current marks drawn on it."""
        if self.reference is None:
            return
        img = self.reference.copy()
        for i, (mx, my) in enumerate(self.marks, 1):
            draw_mark(img, mx, my, i)
        cv2.imwrite(str(self.marked_file), img)


# --------------------------------------------------------------------- UI ----
MARK_COLOR = (0, 255, 255)  # BGR yellow
MARK_GAP = 6                # open radius around the marked pixel: nothing is drawn there
MARK_ARM = 18               # arm length beyond the gap


def draw_label(img, text, org, color=(255, 255, 255), scale=0.55):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = min(max(org[0], 0), img.shape[1] - w - 4)
    y = min(max(org[1], h + 4), img.shape[0] - base - 2)
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_gapped_cross(img, x, y, gap, arm, color, outline=True):
    """Four arms pointing at (x, y) with an empty center, so the target pixel stays visible."""
    arms = [((x - gap - arm, y), (x - gap, y)), ((x + gap, y), (x + gap + arm, y)),
            ((x, y - gap - arm), (x, y - gap)), ((x, y + gap), (x, y + gap + arm))]
    if outline:  # thin dark halo keeps the 1 px arm readable on bright and dark backgrounds
        for a, b in arms:
            cv2.line(img, a, b, (0, 0, 0), 3)
    for a, b in arms:
        cv2.line(img, a, b, color, 1)


def draw_mark(img, x, y, index):
    draw_gapped_cross(img, x, y, MARK_GAP, MARK_ARM, MARK_COLOR)
    # small number tag, pushed off diagonally so it never covers the target area
    draw_label(img, str(index), (x + MARK_GAP + 8, y - MARK_GAP - 8), MARK_COLOR, scale=0.45)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, help="/dev/videoN index (default: auto-detect)")
    ap.add_argument("--ros-topic", help="read frames from this ROS image topic instead of V4L2")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"where marks are kept (default {DEFAULT_DIR})")
    args = ap.parse_args()

    store = Store(args.dir)
    source = RosSource(args.ros_topic) if args.ros_topic else V4L2Source(args.device)
    print(f"Marks file: {store.marks_file}  ({len(store.marks)} loaded)")

    state = {"mouse": None, "overlay": False, "show_marks": True, "clear_armed_until": 0.0}

    def on_mouse(event, x, y, flags, _):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            store.marks.append((x, y))
            store.save_marks()
        elif event == cv2.EVENT_RBUTTONDOWN and store.marks:
            nearest = min(store.marks, key=lambda m: (m[0] - x) ** 2 + (m[1] - y) ** 2)
            store.marks.remove(nearest)
            store.save_marks()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # 1:1 pixels, so window coords == image coords
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            frame = source.read()
            if frame is None:
                frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                draw_label(frame, "waiting for frames...", (20, 40))
            raw = frame.copy()
            view = frame

            ref = store.reference
            if state["overlay"] and ref is not None and ref.shape == view.shape:
                view = cv2.addWeighted(view, 0.5, ref, 0.5, 0)

            if state["show_marks"]:
                for i, (mx, my) in enumerate(store.marks, 1):
                    draw_mark(view, mx, my, i)

            if state["mouse"]:
                x, y = state["mouse"]
                if 0 <= x < view.shape[1] and 0 <= y < view.shape[0]:
                    # full-frame guide lines, but open around the cursor like the marks
                    big = max(view.shape[:2])
                    draw_gapped_cross(view, x, y, MARK_GAP, big, (0, 255, 0), outline=False)
                    b, g, r = raw[y, x]
                    draw_label(view, f"({x}, {y})  rgb {r},{g},{b}", (x + 12, y + 24), (0, 255, 0))

            status = f"{view.shape[1]}x{view.shape[0]}  marks: {len(store.marks)}"
            status += "  overlay: ON" if state["overlay"] else ""
            status += "" if state["show_marks"] else "  marks: HIDDEN"
            if time.time() < state["clear_armed_until"]:
                status += "  -- press c again to CLEAR ALL --"
            draw_label(view, status, (8, 22))
            draw_label(view, "L-click mark | R-click delete | r ref | o overlay | m hide marks"
                             " | c clear | q quit", (8, view.shape[0] - 8), (200, 200, 200))

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r") and source.read() is not None:
                store.save_reference(raw)
                print(f"Reference saved: {store.ref_file} (+ {store.marked_file.name})")
            elif key == ord("m"):
                state["show_marks"] = not state["show_marks"]
            elif key == ord("o"):
                state["overlay"] = not state["overlay"] and store.reference is not None
            elif key == ord("c"):
                if time.time() < state["clear_armed_until"]:
                    store.marks.clear()
                    store.save_marks()
                    state["clear_armed_until"] = 0.0
                else:
                    state["clear_armed_until"] = time.time() + 2.0
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
