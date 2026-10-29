"""YOLO object detection with floor coordinates and stable ids.

Public API (the team contract):
    detect(frame) -> [{"id": int, "label": str, "conf": float,
                       "x": cm, "y": cm, "bbox": [x1, y1, x2, y2]}]
bbox is in raw pixels; x, y in floor cm. The rover itself is excluded.
Returns [] until the tracker has a homography (all 4 corner markers seen once).

Tip: set yolo_model to "yolov8s-worldv2.pt" in config.yaml to detect ANY label in
demo_labels (e.g. "pen", "marker") - COCO's yolov8n has no pen/pencil class.
"""
import math

import numpy as np

from vision.camera import load_config
from vision.tracker import get_tracker


class Detector:
    def __init__(self, tracker=None, cfg=None):
        import torch
        from ultralytics import YOLO

        cfg = cfg or load_config()
        self.tracker = tracker or get_tracker()
        model_name = cfg.get("yolo_model", "yolov8n.pt")
        self.model = YOLO(model_name)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = self.device == 0
        print(f"[detector] {model_name} on {'cuda' if self.device == 0 else 'cpu'}")

        self.imgsz = int(cfg.get("yolo_imgsz", 960))
        self.conf = float(cfg.get("yolo_conf", 0.35))
        self.every_n = max(1, int(cfg.get("yolo_every_n", 1)))
        self.obj_h = float(cfg.get("object_height_cm", 0.0))
        self.rover_excl = float(cfg.get("rover_exclude_cm", 12.0))
        self.match_cm = float(cfg.get("object_match_cm", 15.0))
        self.max_age = int(cfg.get("object_max_age", 30))
        self.margin = float(cfg.get("arena_margin_cm", 10.0))

        labels = list(cfg.get("demo_labels") or [])
        self.classes = None
        if "world" in model_name and labels:
            self.model.set_classes(labels)  # open-vocabulary: names become exactly these labels
        elif labels:
            names = self.model.names
            self.classes = [i for i, n in names.items() if n in labels]
            unknown = set(labels) - set(names.values())
            if unknown:
                print(f"[detector] WARNING not COCO classes (ignored): {sorted(unknown)}")

        self.tracks = []  # [{"id", "label", "x", "y", "age"}]
        self._next_id = 1
        self._frame_i = 0
        self._last = []

    def _assign_ids(self, dets):
        pairs = []
        for di, d in enumerate(dets):
            for ti, t in enumerate(self.tracks):
                if t["label"] == d["label"]:
                    dist = math.hypot(d["x"] - t["x"], d["y"] - t["y"])
                    if dist <= self.match_cm:
                        pairs.append((dist, di, ti))
        pairs.sort()
        used_d, used_t = set(), set()
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            t = self.tracks[ti]
            t.update(x=dets[di]["x"], y=dets[di]["y"], age=0)
            dets[di]["id"] = t["id"]
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        for di, d in enumerate(dets):
            if di not in used_d:
                d["id"] = self._next_id
                self.tracks.append({"id": self._next_id, "label": d["label"],
                                    "x": d["x"], "y": d["y"], "age": 0})
                self._next_id += 1
        return dets

    def detect(self, frame):
        self._frame_i += 1
        if self._frame_i % self.every_n != 0:
            return [dict(d) for d in self._last]

        tr = self.tracker
        tr.observe(frame)
        if not tr.has_homography:
            self._last = []
            return []

        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                               device=self.device, half=self.half, verbose=False)[0]
        if len(r.boxes) == 0:
            self._last = self._assign_ids([])
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        centres = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2])
        floor = tr.parallax(tr.px_to_floor_arr(centres), self.obj_h)

        pose = tr.pose
        dets = []
        for b, c, s, (x, y) in zip(xyxy, cls, conf, floor):
            if not (-self.margin <= x <= tr.W + self.margin and -self.margin <= y <= tr.H + self.margin):
                continue
            if pose is not None and math.hypot(x - pose["x"], y - pose["y"]) < self.rover_excl:
                continue
            dets.append({"label": r.names[int(c)], "conf": float(s), "x": float(x), "y": float(y),
                         "bbox": [float(v) for v in b]})

        dets = self._assign_ids(dets)
        out = [{"id": d["id"], "label": d["label"], "conf": d["conf"],
                "x": d["x"], "y": d["y"], "bbox": d["bbox"]} for d in dets]
        self._last = out
        return [dict(d) for d in out]


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def detect(frame):
    return get_detector().detect(frame)
