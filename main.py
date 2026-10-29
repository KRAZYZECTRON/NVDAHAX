"""rover-ai main loop: typed command -> YOLO -> Cosmos (once) -> drive -> wait again.

Run:
    python main.py                       # sim mode (config.yaml: mode)
    python main.py --mode real           # webcam + Track B vision + ESP32 over UDP
    python main.py --mock                # offline keyword matcher instead of Cosmos (sim tests only)
    python main.py --record demo.mp4     # also save the overlay as a video
    python main.py --headless --command "I need to cut some paper"   # scripted sim run, no window

In the window:
    type a command + Enter     ask Cosmos and drive to the chosen object
    click on the floor         drive to that point (real mode: once all 4 corner markers are seen)
    Esc                        emergency stop / clear the text box
    /goto X Y + Enter          drive to floor point (X, Y) in cm
    /quit + Enter              exit (closing the window also works)
Commands typed in the terminal work too.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from common import load_config

PANEL_W = 420
VIEW_H = 720
GREEN, YELLOW, RED, WHITE, GREY, CYAN = (80, 220, 80), (0, 220, 255), (60, 60, 230), (245, 245, 245), (150, 150, 150), (230, 200, 60)


# ======================================================================= sources
class SimSource:
    """Camera + tracker + detector + sender, all backed by the 2D simulator."""

    def __init__(self, cfg: dict, fixed_dt: float | None = None):
        from sim.sim2d import SimSender, SimWorld

        self.world = SimWorld(cfg)
        self.sender = SimSender(self.world)
        self.fixed_dt = fixed_dt
        self._last = time.monotonic()
        self.arena = (self.world.W, self.world.H)

    def read(self) -> np.ndarray:
        now = time.monotonic()
        self.world.update(self.fixed_dt if self.fixed_dt else now - self._last)
        self._last = now
        return self.world.render()

    def get_pose(self, frame):
        return self.world.get_pose(frame)

    def detect(self, frame):
        return self.world.detect(frame)

    def pixel_to_floor(self, u, v):
        return self.world.to_floor(u, v)

    def floor_to_pixel(self, x, y):
        return self.world.to_px(x, y)

    def close(self):
        self.sender.close()


class RealSource:
    """Overhead webcam + Track B's vision modules + UDP to the ESP32."""

    def __init__(self, cfg: dict):
        from control.drive import UdpSender

        try:
            from vision import detector, tracker
            from vision.camera import open_camera
        except ImportError as e:
            sys.exit(f"Real mode needs Track B's vision package and its dependencies: {e}")
        self.tracker, self.detector = tracker, detector
        self.cap = open_camera(cfg)  # Track B's DirectShow/MJPG setup with locked focus/exposure
        self.sender = UdpSender()
        self.arena = (float(cfg.get("arena_w_cm", 200)), float(cfg.get("arena_h_cm", 150)))

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def get_pose(self, frame):
        return self.tracker.get_pose(frame)

    def detect(self, frame):
        return self.detector.detect(frame)

    def pixel_to_floor(self, u, v):
        return self.tracker.px_to_floor(u, v)   # None until the homography exists

    def floor_to_pixel(self, x, y):
        p = self.tracker.floor_to_px(x, y)
        return None if p is None else (int(p[0]), int(p[1]))

    def close(self):
        self.sender.close()
        self.cap.release()


# ======================================================================= app
class App:
    def __init__(self, args, cfg: dict):
        self.cfg = cfg
        self.args = args
        from ai import cosmos
        from control.drive import Controller

        self.cosmos = cosmos
        self.ctrl = Controller(cfg)
        self.src = SimSource(cfg, fixed_dt=1 / 30 if args.headless else None) if args.mode == "sim" else RealSource(cfg)
        m = cfg.get("mission", {})
        self.stop_short = float(m.get("stop_short_cm", 14))
        self.drive_timeout = float(m.get("drive_timeout_s", 40))

        self.state = "IDLE"            # IDLE | THINKING | DRIVING | ARRIVED | STOPPED
        self.state_t = time.monotonic()
        self.typed = ""
        self.command = ""
        self.reason = "Type what you need and press Enter."
        self.objects: list[dict] = []
        self.target_obj: dict | None = None
        self.target_xy: tuple[float, float] | None = None
        self.path: list[tuple[float, float]] = []
        self.pose: dict | None = None
        self.last_lr = (0, 0)
        self.lost_since: float | None = None
        self.latency: float | None = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future: Future | None = None
        self.inbox: queue.Queue[str] = queue.Queue()
        self.last_detect = 0.0
        self.writer = None
        self.quit = False
        self._scale = 1.0
        self.map_flip = (False, True)   # (flip x, flip y) so the mini-map matches the camera view

    # ------------------------------------------------------------- state
    def set_state(self, s: str) -> None:
        self.state, self.state_t = s, time.monotonic()
        print(f"[{time.strftime('%H:%M:%S')}] state -> {s}")

    def submit(self, text: str, frame: np.ndarray) -> None:
        text = text.strip()
        if not text:
            return
        if text == "/quit":
            self.quit = True
            return
        if text.startswith("/goto"):
            try:
                _, x, y = text.split()
                self.start_drive((float(x), float(y)), label=f"point ({float(x):.0f}, {float(y):.0f})")
            except ValueError:
                self.reason = "Usage: /goto X Y   (floor cm)"
            return
        if self.state == "THINKING":
            return
        self.stop_rover()
        self.command = text
        self.objects = self.src.detect(frame)
        self.last_detect = time.monotonic()
        self.target_obj, self.target_xy = None, None
        self.reason = "Thinking..."
        self.set_state("THINKING")
        print(f"command: {text!r} | objects: {[(o['id'], o['label']) for o in self.objects]}")
        self.future = self.pool.submit(self.cosmos.choose_target, frame.copy(), text, list(self.objects))

    def poll_cosmos(self) -> None:
        if self.state != "THINKING" or self.future is None or not self.future.done():
            return
        info = dict(self.cosmos.last_call_info)
        self.latency = info.get("latency_s")
        try:
            res = self.future.result()
        except Exception as e:
            self.reason = f"Cosmos error: {e}"[:200]
            print(self.reason)
            self.set_state("IDLE")
            return
        self.reason = res["reason"]
        print(f"cosmos ({self.latency or 0:.1f}s): {res}")
        obj = next((o for o in self.objects if o["id"] == res["target_id"]), None)
        if obj is None:
            self.set_state("IDLE")
            return
        self.target_obj = obj
        self.start_drive((obj["x"], obj["y"]), label=obj["label"], approach=True)

    def update_orientation(self) -> None:
        """Work out the floor frame's handedness from the homography.

        Track B's frame runs marker 1 -> 2 (+x) and 1 -> 4 (+y). If that frame has the
        same handedness as the image (y down), +theta is clockwise seen from above, so
        the controller must turn the other way. Assumes the camera image isn't mirrored.
        """
        W, H = self.src.arena
        p0 = self.src.floor_to_pixel(W / 2, H / 2)
        px = self.src.floor_to_pixel(W / 2 + 10, H / 2)
        py = self.src.floor_to_pixel(W / 2, H / 2 + 10)
        if p0 is None or px is None or py is None:
            return
        ax, ay = px[0] - p0[0], px[1] - p0[1]
        bx, by = py[0] - p0[0], py[1] - p0[1]
        det = ax * by - ay * bx
        self.map_flip = (abs(ax) >= abs(ay) and ax < 0, not (abs(by) >= abs(bx) and by > 0))
        if self.ctrl.auto_sign:
            sign = -1 if det > 0 else 1
            if sign != self.ctrl.sign:
                print(f"[main] floor frame is {'clockwise' if det > 0 else 'counter-clockwise'} "
                      f"seen from above -> turn_sign {sign}")
            self.ctrl.sign = sign

    def start_drive(self, xy: tuple[float, float], label: str, approach: bool = False) -> None:
        self.update_orientation()
        tx, ty = xy
        if approach and self.pose is not None:  # stop short of the object instead of ramming it
            dx, dy = tx - self.pose["x"], ty - self.pose["y"]
            d = math.hypot(dx, dy)
            if d > self.stop_short:
                tx, ty = tx - dx / d * self.stop_short, ty - dy / d * self.stop_short
        self.target_xy = (tx, ty)
        if not approach:
            self.target_obj = None
            self.command = f"go to {label}"
        self.path = []
        self.ctrl.reset()
        self.lost_since = None
        self.set_state("DRIVING")

    def stop_rover(self) -> None:
        self.src.sender.stop()
        self.last_lr = (0, 0)

    def control_tick(self) -> None:
        now = time.monotonic()
        if self.state != "DRIVING":
            self.src.sender.send(0, 0)  # keep-alive stop; harmless with the failsafe
            if self.state in ("ARRIVED", "STOPPED") and now - self.state_t > 4:
                self.set_state("IDLE")
            return
        if self.pose is None:
            self.lost_since = self.lost_since or now
        else:
            self.lost_since = None
            if not self.path or math.hypot(self.pose["x"] - self.path[-1][0], self.pose["y"] - self.path[-1][1]) > 1.5:
                self.path.append((self.pose["x"], self.pose["y"]))
        left, right, done = self.ctrl.step(self.pose, self.target_xy)
        if done:
            self.stop_rover()
            self.set_state("ARRIVED")
            return
        if now - self.state_t > self.drive_timeout:
            self.stop_rover()
            self.reason = "Gave up: target not reached in time."
            self.set_state("STOPPED")
            return
        if self.src.sender.send(left, right):
            self.last_lr = (left, right)

    # ------------------------------------------------------------- drawing
    def draw(self, frame: np.ndarray) -> np.ndarray:
        cam = frame.copy()
        for o in self.objects:
            if o.get("bbox"):
                x1, y1, x2, y2 = (int(v) for v in o["bbox"])
                is_t = self.target_obj is not None and o["id"] == self.target_obj["id"]
                col, th = (GREEN, 4) if is_t else (YELLOW, 2)
                cv2.rectangle(cam, (x1, y1), (x2, y2), col, th)
                cv2.putText(cam, f"{o['id']}:{o['label']}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        f2p = self.src.floor_to_pixel
        if self.path and f2p(*self.path[0]) is not None:
            pts = np.array([f2p(x, y) for x, y in self.path], np.int32)
            cv2.polylines(cam, [pts], False, CYAN, 3)
        if self.target_xy and f2p(*self.target_xy) is not None:
            cv2.drawMarker(cam, tuple(int(v) for v in f2p(*self.target_xy)), GREEN, cv2.MARKER_CROSS, 28, 3)

        h, w = cam.shape[:2]
        self._scale = VIEW_H / h
        cam = cv2.resize(cam, (int(w * self._scale), VIEW_H))
        self._cam_w = cam.shape[1]
        panel = self.draw_panel()
        out = np.hstack([cam, panel])
        self.draw_input_bar(out)
        return out

    def draw_panel(self) -> np.ndarray:
        p = np.full((VIEW_H, PANEL_W, 3), (32, 28, 26), np.uint8)
        y = 34
        cv2.putText(p, "Fetch Rover", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        y += 24
        model = self.cosmos.model_name().split("/")[-1]
        if self.cosmos.last_call_info.get("text_only"):
            model += " (text)"
        cv2.putText(p, f"{model} on Nebius Token Factory"[:46], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        if self.cfg.get("cosmos", {}).get("mock") or os.environ.get("COSMOS_MOCK", "") not in ("", "0"):
            cv2.putText(p, "MOCK MODE - NOT COSMOS", (18, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        y += 48
        y = self.text_block(p, "COMMAND", self.command or "-", y, WHITE)
        dots = "." * (1 + int(time.monotonic() * 2) % 3)
        label = "AI IS THINKING" + dots if self.state == "THINKING" else "AI REASONING"
        if self.latency is not None and self.state != "THINKING":
            label += f"  ({self.latency:.1f}s)"
        y = self.text_block(p, label, self.reason, y, YELLOW)
        status = {"IDLE": "Waiting for a command", "THINKING": "Choosing a target...", "DRIVING": "Driving",
                  "ARRIVED": "Arrived!", "STOPPED": "Stopped"}[self.state]
        if self.state == "THINKING":
            status += f" {time.monotonic() - self.state_t:.0f}s"
        if self.state == "DRIVING" and self.lost_since:
            status = "Tracking lost - stopped"
        tgt = self.target_obj["label"] if self.target_obj else ("point" if self.target_xy else "-")
        y = self.text_block(p, "STATUS", f"{status}   target: {tgt}   L,R = {self.last_lr[0]},{self.last_lr[1]}", y, GREEN)
        self.draw_minimap(p, top=max(y + 10, VIEW_H - 300))
        return p

    def text_block(self, img, title, text, y, color) -> int:
        cv2.putText(img, title, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        y += 24
        for line in wrap(text, 34)[:6]:
            cv2.putText(img, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            y += 23
        return y + 14

    def draw_minimap(self, p, top: int) -> None:
        W, H = self.src.arena
        mw = PANEL_W - 36
        s = min(mw / W, (VIEW_H - top - 30) / H)
        ox, oy = 18, top

        fx, fy = self.map_flip

        def m(x, y):
            return int(ox + ((W - x) if fx else x) * s), int(oy + ((H - y) if fy else y) * s)

        cv2.putText(p, "TOP-DOWN MAP (cm)", (ox, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        cv2.rectangle(p, (ox, oy), (int(ox + W * s), int(oy + H * s)), (90, 90, 90), 1)
        for o in self.objects:
            is_t = self.target_obj is not None and o["id"] == self.target_obj["id"]
            cv2.circle(p, m(o["x"], o["y"]), 6 if is_t else 4, GREEN if is_t else YELLOW, -1)
            cv2.putText(p, o["label"], (m(o["x"], o["y"])[0] + 7, m(o["x"], o["y"])[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1)
        if len(self.path) > 1:
            cv2.polylines(p, [np.array([m(x, y) for x, y in self.path], np.int32)], False, CYAN, 2)
        if self.target_xy:
            cv2.drawMarker(p, m(*self.target_xy), GREEN, cv2.MARKER_CROSS, 14, 2)
        if self.pose:
            x, y, th = self.pose["x"], self.pose["y"], self.pose["theta"]
            tri = [m(x + 8 * math.cos(th + a), y + 8 * math.sin(th + a)) for a in (0, 2.5, -2.5)]
            cv2.fillPoly(p, [np.array(tri, np.int32)], WHITE)

    def draw_input_bar(self, out) -> None:
        h = 46
        cv2.rectangle(out, (0, VIEW_H - h), (out.shape[1] - PANEL_W, VIEW_H), (20, 20, 20), -1)
        cursor = "|" if int(time.monotonic() * 2) % 2 == 0 else " "
        hint = self.typed + cursor if self.typed else "Type a command, e.g. 'I need to cut some paper', then Enter"
        cv2.putText(out, "> " + hint, (14, VIEW_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE if self.typed else GREY, 1, cv2.LINE_AA)

    # ------------------------------------------------------------- input
    def on_mouse(self, event, u, v, *_):
        if event != cv2.EVENT_LBUTTONDOWN or self._scale <= 0:
            return
        xy = self.src.pixel_to_floor(u / self._scale, v / self._scale)
        if xy is not None and u < getattr(self, "_cam_w", 0) and v < VIEW_H - 46:
            self.inbox.put(f"/goto {xy[0]:.1f} {xy[1]:.1f}")

    def on_key(self, k: int, frame) -> None:
        if k in (-1, 255):
            return
        if k == 27:  # Esc
            if self.state in ("DRIVING", "THINKING"):
                self.stop_rover()
                self.reason = "Stopped by user."
                self.set_state("STOPPED")
            self.typed = ""
        elif k in (13, 10):
            text, self.typed = self.typed, ""
            self.submit(text, frame)
        elif k in (8, 127):
            self.typed = self.typed[:-1]
        elif 32 <= k <= 126:
            self.typed += chr(k)

    def _stdin_reader(self) -> None:
        for line in sys.stdin:
            self.inbox.put(line.rstrip("\n"))

    # ------------------------------------------------------------- run
    def run(self) -> None:
        win = "rover-ai"
        if not self.args.headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win, self.on_mouse)
            threading.Thread(target=self._stdin_reader, daemon=True).start()
        if self.args.command:
            self.inbox.put(self.args.command)
        view = None
        try:
            while not self.quit:
                frame = self.src.read()
                self.pose = self.src.get_pose(frame)
                while not self.inbox.empty():
                    self.submit(self.inbox.get(), frame)
                if self.state == "IDLE" and time.monotonic() - self.last_detect > 1.0 and not self.args.headless:
                    self.objects = self.src.detect(frame)  # refresh labels on screen while idle
                    self.last_detect = time.monotonic()
                    self.update_orientation()
                self.poll_cosmos()
                self.control_tick()
                view = self.draw(frame)
                if self.args.record:
                    if self.writer is None:
                        self.writer = cv2.VideoWriter(self.args.record, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                                      (view.shape[1], view.shape[0]))
                    self.writer.write(view)
                if self.args.headless:
                    if self.state in ("ARRIVED", "STOPPED") or (self.state == "IDLE" and self.command and self.future and self.future.done()):
                        break
                    if self.state == "THINKING":
                        time.sleep(0.02)  # wait on the Cosmos thread without advancing sim time too fast
                    continue
                cv2.imshow(win, view)
                self.on_key(cv2.waitKey(1), frame)
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.stop_rover()
            self.src.close()
            if self.writer:
                self.writer.release()
            if self.args.snapshot and view is not None:
                cv2.imwrite(self.args.snapshot, view)
            if not self.args.headless:
                cv2.destroyAllWindows()
            self.pool.shutdown(wait=False, cancel_futures=True)
        if self.args.headless:
            print(f"RESULT state={self.state} target={self.target_obj and self.target_obj['label']} reason={self.reason!r}")


def wrap(text: str, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def main() -> None:
    ap = argparse.ArgumentParser(description="rover-ai: command -> Cosmos -> drive")
    ap.add_argument("--mode", choices=["sim", "real"])
    ap.add_argument("--config", help="path to a config.yaml")
    ap.add_argument("--mock", action="store_true", help="offline keyword matcher instead of Cosmos")
    ap.add_argument("--record", help="save the overlay video to this .mp4")
    ap.add_argument("--headless", action="store_true", help="no window (sim tests)")
    ap.add_argument("--command", help="command to run on start")
    ap.add_argument("--snapshot", help="save the last overlay frame as an image")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    args.mode = args.mode or cfg.get("mode", "sim")
    if args.mock:
        os.environ["COSMOS_MOCK"] = "1"
    if args.headless and args.mode != "sim":
        sys.exit("--headless only works in sim mode")
    App(args, cfg).run()


if __name__ == "__main__":
    main()
