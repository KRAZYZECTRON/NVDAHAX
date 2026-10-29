"""Track C: go-to-target controller and UDP sender.

Interface (team contract):
    step(pose, target_xy) -> (L, R, done)

    pose      : {"x": cm, "y": cm, "theta": rad} from the tracker, or None
    target_xy : (x_cm, y_cm) in the same floor frame
    L, R      : ints in -255..255 for the left / right motor
    done      : True once the rover centre is within done_dist_cm of the target

Behaviour: heading error over turn_threshold_deg -> turn in place; otherwise
drive forward with proportional steering, slowing down near the target.
pose None -> (0, 0, False): stop when tracking drops out. Gains: config.yaml.

UdpSender sends the "L,R" ASCII packet to the ESP32 on port 4210.

Test without hardware:  python -m control.drive --selftest
Drive the real rover by hand (no camera): python -m control.drive --spin
"""
from __future__ import annotations

import math
import os
import socket
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config, wrap_angle


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Controller:
    """Stateful turn-then-drive controller (state = turning or not, for hysteresis)."""

    def __init__(self, cfg: dict | None = None):
        c = (cfg or load_config()).get("control", {})
        self.turn_thr = math.radians(c.get("turn_threshold_deg", 20))
        self.turn_exit = math.radians(c.get("turn_exit_deg", 8))
        self.done_dist = float(c.get("done_dist_cm", 8))
        self.fwd = float(c.get("forward_pwm", 170))
        self.turn = float(c.get("turn_pwm", 150))
        self.k = float(c.get("k_heading", 220))
        self.slow_dist = float(c.get("slow_dist_cm", 35))
        self.min_pwm = float(c.get("min_pwm", 120))
        self.max_pwm = float(c.get("max_pwm", 255))
        ts = c.get("turn_sign", "auto")
        self.auto_sign = ts == "auto"   # main.py works it out from the homography
        self.sign = 1 if self.auto_sign or float(ts) >= 0 else -1
        self.turning = True

    def reset(self) -> None:
        self.turning = True

    def _deadband(self, v: float) -> int:
        if abs(v) < 1:
            return 0
        if abs(v) < self.min_pwm:
            v = math.copysign(self.min_pwm, v)
        return int(round(_clamp(v, -self.max_pwm, self.max_pwm)))

    def step(self, pose: dict | None, target_xy) -> tuple[int, int, bool]:
        if pose is None or target_xy is None:
            return 0, 0, False
        dx = float(target_xy[0]) - pose["x"]
        dy = float(target_xy[1]) - pose["y"]
        dist = math.hypot(dx, dy)
        if dist <= self.done_dist:
            self.turning = True
            return 0, 0, True

        err = wrap_angle(math.atan2(dy, dx) - pose["theta"])  # + = target is to the left (CCW)

        # hysteresis: start turning above turn_thr, stop turning below turn_exit
        if abs(err) > self.turn_thr:
            self.turning = True
        elif abs(err) < self.turn_exit:
            self.turning = False

        if self.turning:
            # slow the spin down as we line up so we don't overshoot
            s = self.turn * _clamp(abs(err) / math.radians(45), 0.6, 1.0)
            s = math.copysign(s, err) * self.sign
            return self._deadband(-s), self._deadband(s), False  # CCW: left back, right forward

        v = self.fwd * _clamp(dist / self.slow_dist, 0.55, 1.0)
        w = self.k * err * self.sign
        left, right = v - w, v + w
        # never let steering reverse a wheel while driving forward
        left, right = max(left, 0.0), max(right, 0.0)
        return self._deadband(left), self._deadband(right), False


_default: Controller | None = None


def step(pose: dict | None, target_xy) -> tuple[int, int, bool]:
    """Module-level contract function; uses one shared Controller built from config.yaml."""
    global _default
    if _default is None:
        _default = Controller()
    return _default.step(pose, target_xy)


def reset() -> None:
    """Call when a new target is chosen."""
    if _default is not None:
        _default.reset()


class UdpSender:
    """Sends "L,R" packets to the ESP32. Rate-limit with send_hz; the ESP32 stops after 300 ms of silence."""

    def __init__(self, ip: str | None = None, port: int | None = None, send_hz: float | None = None):
        r = load_config().get("rover", {})
        self.addr = (ip or r.get("ip", "192.168.43.100"), int(port or r.get("port", 4210)))
        self.period = 1.0 / float(send_hz or r.get("send_hz", 15))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last = 0.0
        self.last_cmd = (0, 0)

    def send(self, left: int, right: int, force: bool = False) -> bool:
        """Send if at least one period has passed (or force). Returns True if a packet went out."""
        now = time.monotonic()
        if not force and now - self._last < self.period:
            return False
        left = int(_clamp(left, -255, 255))
        right = int(_clamp(right, -255, 255))
        self.sock.sendto(f"{left},{right}".encode("ascii"), self.addr)
        self._last = now
        self.last_cmd = (left, right)
        return True

    def stop(self) -> None:
        for _ in range(3):  # UDP can drop a packet; send stop a few times
            self.send(0, 0, force=True)
            time.sleep(0.02)

    def close(self) -> None:
        self.stop()
        self.sock.close()


# ---------------------------------------------------------------- self tests
def _selftest() -> None:
    c = Controller()
    cases = [
        ({"x": 0, "y": 0, "theta": 0.0}, (100, 0), "straight ahead -> both forward"),
        ({"x": 0, "y": 0, "theta": 0.0}, (0, 100), "target 90 deg left -> spin CCW (L<0<R)"),
        ({"x": 0, "y": 0, "theta": 0.0}, (0, -100), "target 90 deg right -> spin CW (L>0>R)"),
        ({"x": 0, "y": 0, "theta": 0.0}, (100, 15), "slightly left -> R > L"),
        ({"x": 0, "y": 0, "theta": 0.0}, (5, 0), "inside 8 cm -> done"),
        (None, (100, 0), "lost tracking -> stop"),
    ]
    for pose, tgt, note in cases:
        c.reset()
        c.turning = False if note.startswith(("straight", "slightly")) else True
        print(f"{note:45s} -> {c.step(pose, tgt)}")


def _spin() -> None:
    """Hardware smoke test: forward, back, left, right for 1 s each, then stop."""
    s = UdpSender()
    print("sending to", s.addr)
    try:
        for name, (l, r) in [("forward", (160, 160)), ("back", (-160, -160)), ("left", (-150, 150)), ("right", (150, -150))]:
            print(name)
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                s.send(l, r)
                time.sleep(0.01)
            s.stop()
            time.sleep(0.5)
    finally:
        s.close()


if __name__ == "__main__":
    if "--spin" in sys.argv:
        _spin()
    else:
        _selftest()
