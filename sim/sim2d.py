"""Track C: fake top-down world with the SAME interfaces as the real hardware.

    world = SimWorld()
    frame = world.render()            # like a camera frame (BGR image)
    pose  = world.get_pose(frame)     # like vision.tracker.get_pose
    objs  = world.detect(frame)       # like vision.detector.detect
    sender = SimSender(world)         # like control.drive.UdpSender
    sender.send(L, R); world.update(dt)

The physics is a simple differential-drive model with a motor deadband, command
latency, pose noise, tracking dropouts and the ESP32's 300 ms failsafe, so the
controller tuned here should transfer to the real rover with small gain changes.

Fake ESP32 for testing UDP code (teleop, UdpSender) with no hardware:
    python -m sim.sim2d --udp          # then send "L,R" packets to 127.0.0.1:4210
"""
from __future__ import annotations

import math
import os
import random
import socket
import sys
import time
from collections import deque

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config, wrap_angle

# COCO labels that yolov8n can detect and that make sense for fetch commands.
# Note: COCO has no "pen"/"pencil" class.
OBJECT_KINDS = {
    "cup": ("circle", 4.5, (40, 90, 200)),
    "bottle": ("rect", (7, 22), (60, 160, 60)),
    "scissors": ("rect", (6, 18), (90, 90, 90)),
    "cell phone": ("rect", (7.5, 15), (30, 30, 30)),
    "book": ("rect", (15, 21), (150, 70, 40)),
    "banana": ("rect", (4, 18), (40, 210, 230)),
    "apple": ("circle", 4.0, (40, 40, 200)),
    "sports ball": ("circle", 6.5, (220, 140, 20)),
    "remote": ("rect", (4.5, 16), (80, 60, 60)),
    "teddy bear": ("circle", 9.0, (60, 110, 160)),
}


class SimWorld:
    def __init__(self, cfg: dict | None = None, seed: int | None = None, objects: list[tuple[str, float, float]] | None = None):
        cfg = cfg or load_config()
        s = cfg.get("sim", {})
        self.W = float(cfg.get("arena_w_cm", 200))  # Track B's arena keys
        self.H = float(cfg.get("arena_h_cm", 150))
        self.wheel_base = float(s.get("wheel_base_cm", 13.5))
        self.vmax = float(s.get("max_wheel_speed_cm_s", 35))
        self.deadband = float(s.get("deadband_pwm", 70))
        self.pose_noise = float(s.get("pose_noise_cm", 0.4))
        self.heading_noise = math.radians(float(s.get("heading_noise_deg", 1.0)))
        self.dropout = float(s.get("dropout_prob", 0.01))
        self.latency = float(s.get("latency_s", 0.06))
        self.ppc = float(s.get("px_per_cm", 4.5))
        self.margin = 30
        self.rng = random.Random(s.get("seed", 7) if seed is None else seed)

        self.t = 0.0
        self.x, self.y, self.theta = self.W * 0.5, self.H * 0.15, math.pi / 2
        self.cmd_queue: deque = deque()  # (time_sent, L, R)
        self.applied = (0, 0)
        self.last_rx = -1e9
        self.trail: list[tuple[float, float]] = []

        if objects is None:
            objects = self._random_objects(int(s.get("num_objects", 5)))
        self.objects = [{"id": i, "label": l, "x": float(x), "y": float(y)} for i, (l, x, y) in enumerate(objects)]

    # ------------------------------------------------------------ setup
    def _random_objects(self, n: int) -> list[tuple[str, float, float]]:
        labels = self.rng.sample(list(OBJECT_KINDS), k=min(n, len(OBJECT_KINDS)))
        placed: list[tuple[str, float, float]] = []
        for lab in labels:
            for _ in range(200):
                x = self.rng.uniform(20, self.W - 20)
                y = self.rng.uniform(self.H * 0.35, self.H - 15)
                if all(math.hypot(x - px, y - py) > 28 for _, px, py in placed):
                    placed.append((lab, x, y))
                    break
        return placed

    # ------------------------------------------------------------ geometry
    def to_px(self, x: float, y: float) -> tuple[int, int]:
        """Floor cm -> image pixel (y up on the floor = up on screen)."""
        return int(self.margin + x * self.ppc), int(self.margin + (self.H - y) * self.ppc)

    def to_floor(self, u: float, v: float) -> tuple[float, float]:
        """Image pixel -> floor cm (used for click-to-drive)."""
        return (u - self.margin) / self.ppc, self.H - (v - self.margin) / self.ppc

    @property
    def size_px(self) -> tuple[int, int]:
        return int(self.W * self.ppc + 2 * self.margin), int(self.H * self.ppc + 2 * self.margin)

    # ------------------------------------------------------------ physics
    def receive(self, left: int, right: int) -> None:
        self.cmd_queue.append((self.t, int(left), int(right)))
        self.last_rx = self.t

    def _wheel(self, pwm: int) -> float:
        return 0.0 if abs(pwm) < self.deadband else (pwm / 255.0) * self.vmax

    def update(self, dt: float) -> None:
        dt = max(0.0, min(dt, 0.2))
        self.t += dt
        while self.cmd_queue and self.cmd_queue[0][0] <= self.t - self.latency:
            _, l, r = self.cmd_queue.popleft()
            self.applied = (l, r)
        if self.t - self.last_rx > 0.3:  # ESP32 failsafe
            self.applied = (0, 0)
        vl, vr = self._wheel(self.applied[0]), self._wheel(self.applied[1])
        v, w = (vl + vr) / 2.0, (vr - vl) / self.wheel_base
        self.theta = wrap_angle(self.theta + w * dt)
        self.x = min(max(self.x + v * math.cos(self.theta) * dt, 5), self.W - 5)
        self.y = min(max(self.y + v * math.sin(self.theta) * dt, 5), self.H - 5)
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 1.0:
            self.trail.append((self.x, self.y))

    # ------------------------------------------------------------ fake sensors
    def get_pose(self, frame=None) -> dict | None:
        if self.rng.random() < self.dropout:
            return None
        return {
            "x": self.x + self.rng.gauss(0, self.pose_noise),
            "y": self.y + self.rng.gauss(0, self.pose_noise),
            "theta": wrap_angle(self.theta + self.rng.gauss(0, self.heading_noise)),
        }

    def detect(self, frame=None) -> list[dict]:
        out = []
        for o in self.objects:
            kind = OBJECT_KINDS[o["label"]]
            half = (kind[1], kind[1]) if kind[0] == "circle" else (kind[1][0] / 2, kind[1][1] / 2)
            x1, y1 = self.to_px(o["x"] - half[0], o["y"] + half[1])
            x2, y2 = self.to_px(o["x"] + half[0], o["y"] - half[1])
            out.append({
                "id": o["id"], "label": o["label"], "conf": round(self.rng.uniform(0.6, 0.95), 2),
                "x": o["x"], "y": o["y"], "bbox": [x1 - 4, y1 - 4, x2 + 4, y2 + 4],
            })
        return out

    # ------------------------------------------------------------ fake camera
    def render(self) -> np.ndarray:
        w, h = self.size_px
        img = np.full((h, w, 3), (196, 212, 222), np.uint8)  # floor
        for gx in range(0, int(self.W) + 1, 25):
            cv2.line(img, self.to_px(gx, 0), self.to_px(gx, self.H), (180, 196, 206), 1)
        for gy in range(0, int(self.H) + 1, 25):
            cv2.line(img, self.to_px(0, gy), self.to_px(self.W, gy), (180, 196, 206), 1)
        for i, (cx, cy) in enumerate([(0, 0), (self.W, 0), (self.W, self.H), (0, self.H)]):
            u, v = self.to_px(cx, cy)
            cv2.rectangle(img, (u - 12, v - 12), (u + 12, v + 12), (0, 0, 0), -1)
            cv2.rectangle(img, (u - 6, v - 6), (u + 6, v + 6), (255, 255, 255), -1)
        for o in self.objects:
            self._draw_object(img, o)
        self._draw_rover(img)
        return img

    def _draw_object(self, img: np.ndarray, o: dict) -> None:
        shape, size, color = OBJECT_KINDS[o["label"]]
        u, v = self.to_px(o["x"], o["y"])
        if shape == "circle":
            r = int(size * self.ppc)
            cv2.circle(img, (u, v), r, color, -1)
            cv2.circle(img, (u, v), r, (20, 20, 20), 2)
        else:
            hw, hh = int(size[0] * self.ppc / 2), int(size[1] * self.ppc / 2)
            cv2.rectangle(img, (u - hw, v - hh), (u + hw, v + hh), color, -1)
            cv2.rectangle(img, (u - hw, v - hh), (u + hw, v + hh), (20, 20, 20), 2)
        cv2.putText(img, o["label"], (u - 25, v + int(14 * self.ppc)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

    def _draw_rover(self, img: np.ndarray) -> None:
        c, s = math.cos(self.theta), math.sin(self.theta)

        def pt(fx: float, fy: float) -> tuple[int, int]:  # rover frame (forward, left) -> px
            return self.to_px(self.x + fx * c - fy * s, self.y + fx * s + fy * c)

        body = np.array([pt(10, 8), pt(10, -8), pt(-8, -8), pt(-8, 8)], np.int32)
        cv2.fillPoly(img, [body], (235, 235, 235))
        cv2.polylines(img, [body], True, (40, 40, 40), 2)
        for side in (9.5, -9.5):  # wheels
            wheel = np.array([pt(3.5, side - 1.5), pt(3.5, side + 1.5), pt(-3.5, side + 1.5), pt(-3.5, side - 1.5)], np.int32)
            cv2.fillPoly(img, [wheel], (30, 30, 30))
        marker = np.array([pt(4, 4), pt(4, -4), pt(-4, -4), pt(-4, 4)], np.int32)
        cv2.fillPoly(img, [marker], (0, 0, 0))
        cv2.arrowedLine(img, pt(-3, 0), pt(4, 0), (255, 255, 255), 2, tipLength=0.4)


class SimSender:
    """Drop-in for control.drive.UdpSender that delivers commands to a SimWorld."""

    def __init__(self, world: SimWorld, send_hz: float | None = None):
        self.world = world
        self.period = 1.0 / float(send_hz or load_config().get("rover", {}).get("send_hz", 15))
        self._last = -1e9
        self.last_cmd = (0, 0)

    def send(self, left: int, right: int, force: bool = False) -> bool:
        if not force and self.world.t - self._last < self.period:
            return False
        self.world.receive(left, right)
        self._last = self.world.t
        self.last_cmd = (int(left), int(right))
        return True

    def stop(self) -> None:
        self.send(0, 0, force=True)

    def close(self) -> None:
        self.stop()


def run_fake_esp32(port: int = 4210) -> None:
    """Listen for "L,R" UDP packets like the ESP32 and show the simulated rover."""
    world = SimWorld()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    print(f"fake ESP32 listening on UDP {port}; send 'L,R' to 127.0.0.1:{port}. Esc quits.")
    last = time.monotonic()
    while True:
        try:
            while True:
                data, _ = sock.recvfrom(64)
                l, r = (int(v) for v in data.decode("ascii").strip().split(","))
                world.receive(max(-255, min(255, l)), max(-255, min(255, r)))
        except BlockingIOError:
            pass
        except ValueError:
            print("bad packet:", data)
        now = time.monotonic()
        world.update(now - last)
        last = now
        img = world.render()
        cv2.putText(img, f"applied L,R = {world.applied}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow("fake ESP32", img)
        if cv2.waitKey(15) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--udp" in sys.argv:
        run_fake_esp32(int(load_config().get("rover", {}).get("port", 4210)))
    else:
        w = SimWorld()
        cv2.imwrite("sim_frame.png", w.render())
        print("objects:", w.detect())
        print("pose:", w.get_pose())
        print("wrote sim_frame.png")
