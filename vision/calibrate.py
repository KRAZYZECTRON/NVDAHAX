"""One-off lens calibration with the printed checkerboard -> vision/camera_calib.npz.

Run at the SAME resolution and focus setting you'll use for tracking.
Hold the board (taped to something flat) 0.5-1.5 m from the camera. Take 15-25
shots: tilted at different angles, and covering the corners/edges of the frame.

Keys:  SPACE capture (only when the board is found)   c calibrate + save   q quit
Usage: python -m vision.calibrate [--cols 9 --rows 6 --square-mm 25]
"""
import argparse

import cv2
import numpy as np

from vision.camera import load_config, open_camera, repo_path


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-mm", type=float, default=25.0, help="measure a printed square")
    ap.add_argument("--cam", default=None, help="camera index or stream URL")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm

    cap = open_camera(cfg, index=args.cam)
    obj_pts, img_pts, size = [], [], None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        # Fast preview detection on a half-size image.
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        found, corners = cv2.findChessboardCorners(
            small, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern, corners * 2, found)
        cv2.putText(vis, f"captured: {len(img_pts)}  (SPACE capture, c calibrate, q quit)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibrate", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            ok_full, full = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if ok_full:
                full = cv2.cornerSubPix(gray, full, (11, 11), (-1, -1), term)
                obj_pts.append(objp)
                img_pts.append(full)
                print(f"captured {len(img_pts)}")
        if key == ord("c"):
            if len(img_pts) < 10:
                print("need at least 10 captures (15-25 is better)")
                continue
            rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
            out = repo_path(cfg.get("calib_file", "vision/camera_calib.npz"))
            np.savez(out, K=K, dist=dist, size=np.array(size), rms=rms)
            print(f"RMS reprojection error: {rms:.3f} px  (good < 0.5, OK < 1.0)")
            print("K =\n", K, "\ndist =", dist.ravel())
            print("saved", out)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
