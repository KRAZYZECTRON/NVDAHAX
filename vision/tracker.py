"""Rover pose from the overhead camera: ArUco corner markers -> floor homography,
rover marker (ID 0) -> x, y (cm) and heading theta (rad), with parallax correction.

Public API (the team contract):
    get_pose(frame) -> {"x": cm, "y": cm, "theta": rad} | None
    px_to_floor(u, v) -> (x_cm, y_cm) | None      (None until the homography exists)
    floor_to_px(x_cm, y_cm) -> (u, v) | None

Floor frame: origin at marker 1's centre, +x toward marker 2, +y toward marker 4.
theta = atan2(dy, dx) in that frame (0 = facing +x).
Pixel coordinates are always RAW camera pixels (what cv2 draws on / mouse clicks give).
"""
import math
import os

import cv2
import numpy as np

from vision.camera import load_config, repo_path

ROVER_ID = 0
CORNER_IDS = (1, 2, 3, 4)


class Tracker:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        self.cfg = cfg
        self.W = float(cfg.get("arena_w_cm", 200.0))
        self.H = float(cfg.get("arena_h_cm", 150.0))
        self.cam_h = float(cfg.get("camera_height_cm", 200.0))
        self.marker_h = float(cfg.get("marker_height_cm", 0.0))
        self.alpha = float(cfg.get("pose_ema_alpha", 0.6))
        self.max_missed = int(cfg.get("pose_max_missed", 3))
        self.drift_px = float(cfg.get("homography_drift_px", 3.0))
        self.h_file = repo_path(cfg.get("homography_file", "vision/homography.npz"))

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

        self.K = self.dist = None
        self.calib_size = None
        calib = repo_path(cfg.get("calib_file", "vision/camera_calib.npz"))
        if os.path.exists(calib):
            d = np.load(calib)
            self.K, self.dist = d["K"], d["dist"]
            self.calib_size = tuple(int(v) for v in d["size"])
            print(f"[tracker] lens calibration loaded ({self.calib_size[0]}x{self.calib_size[1]})")
        else:
            print("[tracker] no lens calibration - using raw pixels (expect error near edges)")

        self.Hm = None          # undistorted px -> floor cm
        self.Hm_inv = None
        self.h_src = None       # corner centres (undistorted px) used for Hm
        self.h_version = 0      # bumps whenever Hm changes (for overlay caches)
        self.nadir = None       # floor point straight under the camera
        self.frame_size = None
        self._load_homography()

        self.markers = {}       # id -> 4x2 RAW px corners, last processed frame
        self.pose = None
        self.missed = 0
        self._last_frame = None

    # ---------- geometry helpers ----------
    def _undistort(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if self.K is None:
            return pts.reshape(-1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def _distort(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.K is None:
            return pts
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        xyz = np.column_stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy, np.ones(len(pts))])
        out, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)

    def px_to_floor_arr(self, pts):
        """Nx2 raw px -> Nx2 floor cm (points on the floor, no parallax)."""
        if self.Hm is None:
            return None
        und = self._undistort(pts).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(und, self.Hm).reshape(-1, 2)

    def floor_to_px_arr(self, pts):
        """Nx2 floor cm -> Nx2 raw px."""
        if self.Hm_inv is None:
            return None
        und = cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), self.Hm_inv)
        return self._distort(und.reshape(-1, 2))

    def parallax(self, pts, height_cm):
        """Floor-homography points of something height_cm above the floor -> true floor position."""
        pts = np.asarray(pts, np.float64)
        if height_cm <= 0 or self.nadir is None:
            return pts
        k = (self.cam_h - height_cm) / self.cam_h
        return self.nadir + (pts - self.nadir) * k

    def px_to_floor(self, u, v):
        p = self.px_to_floor_arr([[u, v]])
        return None if p is None else (float(p[0, 0]), float(p[0, 1]))

    def floor_to_px(self, x, y):
        p = self.floor_to_px_arr([[x, y]])
        return None if p is None else (float(p[0, 0]), float(p[0, 1]))

    @property
    def has_homography(self):
        return self.Hm is not None

    # ---------- homography ----------
    def _set_homography(self, src):
        dst = np.array([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]], np.float64)
        Hm, _ = cv2.findHomography(src, dst)
        if Hm is None:
            return
        self.Hm, self.Hm_inv, self.h_src = Hm, np.linalg.inv(Hm), src
        self.h_version += 1
        self._update_nadir()
        np.savez(self.h_file, H=Hm, src=src, W=self.W, Hh=self.H)
        print(f"[tracker] homography updated (v{self.h_version})")

    def _load_homography(self):
        if not os.path.exists(self.h_file):
            return
        d = np.load(self.h_file)
        if abs(float(d["W"]) - self.W) > 1e-6 or abs(float(d["Hh"]) - self.H) > 1e-6:
            print("[tracker] saved homography is for a different arena size - ignoring")
            return
        self.Hm, self.h_src = d["H"], d["src"]
        self.Hm_inv = np.linalg.inv(self.Hm)
        self.h_version += 1
        print("[tracker] loaded saved homography (re-checked live when corners are visible)")

    def _update_nadir(self):
        if self.Hm is None or self.frame_size is None:
            return
        if self.K is not None:
            c = np.array([[self.K[0, 2], self.K[1, 2]]])  # principal point, undistorted px
        else:
            c = np.array([[self.frame_size[0] / 2, self.frame_size[1] / 2]])
        self.nadir = cv2.perspectiveTransform(c.reshape(-1, 1, 2), self.Hm).reshape(2)

    def reset_homography(self):
        """Force a recompute on the next frame where all 4 corners are visible."""
        self.Hm = self.Hm_inv = self.h_src = None
        self.nadir = None

    def _check_homography(self):
        if not all(i in self.markers for i in CORNER_IDS):
            return
        src = self._undistort(np.array([self.markers[i].mean(axis=0) for i in CORNER_IDS]))
        if self.h_src is None or np.max(np.linalg.norm(src - self.h_src, axis=1)) > self.drift_px:
            self._set_homography(src)

    # ---------- per-frame ----------
    def observe(self, frame):
        """Detect markers once per frame (safe to call from get_pose and detect)."""
        if frame is self._last_frame:
            return
        self._last_frame = frame

        h, w = frame.shape[:2]
        if self.frame_size != (w, h):
            self.frame_size = (w, h)
            if self.calib_size is not None and self.calib_size != (w, h):
                print(f"[tracker] WARNING calibration is {self.calib_size}, frames are {(w, h)}"
                      " - ignoring calibration; re-run calibrate.py at this resolution")
                self.K = self.dist = None
            self._update_nadir()

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco.detectMarkers(gray)
        self.markers = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                self.markers[int(i)] = c.reshape(4, 2)

        self._check_homography()
        self._update_pose()

    def _update_pose(self):
        if self.Hm is None or ROVER_ID not in self.markers:
            self.missed += 1
            if self.missed >= self.max_missed:
                self.pose = None
            return
        self.missed = 0

        c = self.markers[ROVER_ID]           # TL, TR, BR, BL; TL->TR is the marker's "up" edge
        centre_px = c.mean(axis=0)
        front_px = (c[0] + c[1]) / 2
        pts = self.px_to_floor_arr([centre_px, front_px])
        pts = self.parallax(pts, self.marker_h)
        x, y = pts[0]
        theta = math.atan2(pts[1, 1] - y, pts[1, 0] - x)

        if self.pose is None or self.alpha >= 1.0:
            self.pose = {"x": float(x), "y": float(y), "theta": float(theta)}
            return
        a = self.alpha
        p = self.pose
        dth = math.atan2(math.sin(theta - p["theta"]), math.cos(theta - p["theta"]))
        th = p["theta"] + a * dth
        self.pose = {
            "x": float(p["x"] + a * (x - p["x"])),
            "y": float(p["y"] + a * (y - p["y"])),
            "theta": float(math.atan2(math.sin(th), math.cos(th))),
        }

    def get_pose(self, frame):
        self.observe(frame)
        return None if self.pose is None else dict(self.pose)


# ---------- module-level wrappers (the contract) ----------
_tracker = None


def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = Tracker()
    return _tracker


def get_pose(frame):
    return get_tracker().get_pose(frame)


def px_to_floor(u, v):
    return get_tracker().px_to_floor(u, v)


def floor_to_px(x_cm, y_cm):
    return get_tracker().floor_to_px(x_cm, y_cm)
