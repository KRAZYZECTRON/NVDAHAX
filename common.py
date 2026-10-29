"""Small helpers shared by every module: config loading and angle maths."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
_CACHE: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (cached). Pass a path to load a different file."""
    global _CACHE
    if path is None and _CACHE is not None:
        return _CACHE
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path is None:
        _CACHE = cfg
    return cfg


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
