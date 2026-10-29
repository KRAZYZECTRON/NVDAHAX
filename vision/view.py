"""Live debug viewer: corner markers, floor grid, rover pose, YOLO objects, FPS.
Click anywhere to print that pixel's floor cm.

Keys:  q quit   h recompute homography   p camera driver settings   s save snapshot
       y toggle YOLO
Usage: python -m vision.view [--cam 1] [--no-yolo] [--width 1280 --height 720]
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

from vision.camera import load_config, open_camera, open_settings_dialog
from vision.tracker import CORNER_IDS, ROVER_ID, get_tracker

GRID_CM = 25
SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def build_grid(tr):
    lines = []
    xs = list(np.arange(0, tr.W, GRID_CM)) + [tr.W]
    ys = list(np.arange(0, tr.H, GRID_CM)) + [tr.H]
    for x in xs:
        pts = np.column_stack([np.full(20, x), np.linspace(0, tr.H, 20)])
        lines.append((tr.floor_to_px_arr(pts), x in (0, tr.W)))
    for y in ys:
        pts = np.column_stack([np.linspace(0, tr.W, 20), np.full(20, y)])
        lines.append((tr.floor_to_px_arr(pts), y in (0, tr.H)))
    return lines


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default=None, help="camera index or stream URL")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--no-yolo", action="store_true")
    args = ap.parse_args()

    cap = open_camera(cfg, index=args.cam, width=args.width, height=args.height)
    tr = get_tracker()
    det = None
    use_yolo = not args.no_yolo
    if use_yolo:
        from vision.detector import get_detector
        det = get_detector()

    click = {"px": None, "floor": None}

    def on_mouse(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            f = tr.px_to_floor(u, v)
            click["px"], click["floor"] = (u, v), f
            if f is None:
                print(f"px ({u}, {v}) -> no homography yet")
            else:
                print(f"px ({u}, {v}) -> floor ({f[0]:.1f}, {f[1]:.1f}) cm")

    win = "track B view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win, on_mouse)
    grid, grid_ver = None, -1
    fps, t_prev = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            break

        pose = tr.get_pose(frame)
        objs = det.detect(frame) if (use_yolo and det) else []
        vis = frame.copy()

        # Floor grid + arena outline.
        if tr.has_homography:
            if grid_ver != tr.h_version:
                grid, grid_ver = build_grid(tr), tr.h_version
            for pts, border in grid:
                cv2.polylines(vis, [pts.astype(np.int32)], False,
                              (0, 200, 255) if border else (90, 90, 90), 2 if border else 1, cv2.LINE_AA)

        # Markers.
        for mid, c in tr.markers.items():
            col = (0, 255, 0) if mid in CORNER_IDS else (255, 0, 255) if mid == ROVER_ID else (128, 128, 128)
            cv2.polylines(vis, [c.astype(np.int32)], True, col, 2)
            cv2.putText(vis, f"ID{mid}", tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # Rover pose arrow (drawn at its true floor position).
        if pose is not None:
            p0 = tr.floor_to_px(pose["x"], pose["y"])
            p1 = tr.floor_to_px(pose["x"] + 20 * math.cos(pose["theta"]),
                                pose["y"] + 20 * math.sin(pose["theta"]))
            p0, p1 = tuple(int(v) for v in p0), tuple(int(v) for v in p1)
            cv2.circle(vis, p0, 8, (255, 0, 255), -1)
            cv2.arrowedLine(vis, p0, p1, (255, 0, 255), 3, tipLength=0.3)
            cv2.putText(vis, f"rover ({pose['x']:.1f}, {pose['y']:.1f}) cm  "
                             f"{math.degrees(pose['theta']):.0f} deg",
                        (p0[0] + 12, p0[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Objects.
        for o in objs:
            x1, y1, x2, y2 = (int(v) for v in o["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(vis, f"#{o['id']} {o['label']} {o['conf']:.2f}", (x1, y1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(vis, f"({o['x']:.0f}, {o['y']:.0f}) cm", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Last click.
        if click["px"] is not None:
            cv2.drawMarker(vis, click["px"], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            if click["floor"] is not None:
                cv2.putText(vis, f"({click['floor'][0]:.1f}, {click['floor'][1]:.1f})",
                            (click["px"][0] + 10, click["px"][1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Status line.
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        corners_seen = sum(i in tr.markers for i in CORNER_IDS)
        status = (f"FPS {fps:.1f} | corners {corners_seen}/4 | "
                  f"homography {'OK' if tr.has_homography else 'MISSING'} | "
                  f"rover {'OK' if pose else 'None'} | objects {len(objs)} | "
                  f"YOLO {'on' if use_yolo else 'off'}")
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("h"):
            tr.reset_homography()
            print("homography reset - will recompute when all 4 corners are visible")
        elif key == ord("p"):
            open_settings_dialog(cap)
        elif key == ord("s"):
            os.makedirs(SNAP_DIR, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(os.path.join(SNAP_DIR, f"raw_{stamp}.jpg"), frame)
            cv2.imwrite(os.path.join(SNAP_DIR, f"view_{stamp}.jpg"), vis)
            print("saved snapshots to", SNAP_DIR)
        elif key == ord("y"):
            if det is None:
                from vision.detector import get_detector
                det = get_detector()
            use_yolo = not use_yolo

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
