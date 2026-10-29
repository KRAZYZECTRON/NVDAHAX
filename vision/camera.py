"""Shared helpers: config loading and opening the overhead webcam (Windows / DirectShow)."""
import os

import cv2
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(p):
    """Resolve a config path relative to the repo root."""
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def open_camera(cfg=None, index=None, width=None, height=None):
    cfg = cfg or load_config()
    idx = cfg.get("camera_index", 0) if index is None else index
    w = width or cfg.get("camera_width", 1920)
    h = height or cfg.get("camera_height", 1080)

    if isinstance(idx, str) and not idx.isdigit():
        # Network stream (phone/iPad camera app). Resolution/focus/exposure are set in the app.
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open stream {idx}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[camera] stream {idx} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return cap

    idx = int(idx)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {idx}")
    # FOURCC must be set before the resolution for MJPG to stick on DirectShow.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, cfg.get("camera_fps", 30))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    focus = cfg.get("camera_focus")
    if focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    exposure = cfg.get("camera_exposure")
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual (ignored by some drivers)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"[camera] index={idx} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} fourcc={fourcc_str} "
          f"fps(reported)={cap.get(cv2.CAP_PROP_FPS):.0f}")
    return cap


def open_settings_dialog(cap):
    """Opens the driver's own settings window (focus/exposure/gain sliders)."""
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
