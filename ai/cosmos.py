"""Track C: ask NVIDIA Cosmos3-Super-Reasoner (on Nebius Token Factory) which object to fetch.

Interface (team contract):
    choose_target(frame, command, objects) -> {"target_id": int, "reason": str}

    frame   : BGR image (numpy array) from the overhead camera
    command : what the user typed, e.g. "bring me something to write with"
    objects : detector output, [{"id", "label", "conf", "x", "y", "bbox"}, ...]

target_id is one of the ids in `objects`, or -1 when nothing fits (the reason
then says why). Called ONCE per command, never inside the control loop.

The API key is read from NEBIUS_API_KEY. NEBIUS_BASE_URL and COSMOS_MODEL
override config.yaml if set. COSMOS_MOCK=1 uses an offline keyword matcher.

Any vision or text model on Token Factory works: with cosmos.send_image "auto",
a model that rejects images is retried with the object list only (text).

Quick test with a photo (first-30-minutes task):
    python -m ai.cosmos photo.jpg "bring me something to write with" --labels cup,scissors,pen
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):  # allow `python ai/cosmos.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config

# Filled after every call so main.py can show latency without changing the contract.
last_call_info: dict = {}
_TEXT_ONLY: set[str] = set()  # models that rejected an image this session

SYSTEM_PROMPT = (
    "You are the brain of a small fetch robot. A camera looks straight down at a floor; "
    "you get the list of objects it detected (and the picture, when available). "
    "The user asks for something. Pick the ONE object from the numbered list that best "
    "satisfies the request, using common sense about what objects are used for. "
    "Only choose ids from the list. If nothing in the list fits, use -1. "
    "Think briefly. Then reply with ONLY a JSON object, no markdown, in this exact form: "
    '{"target_id": <int>, "reason": "<one short sentence a person would understand>"}'
)


def _settings() -> dict:
    cfg = load_config().get("cosmos", {})
    return {
        "base_url": os.environ.get("NEBIUS_BASE_URL") or cfg.get("base_url", "https://api.tokenfactory.nebius.com/v1/"),
        "model": os.environ.get("COSMOS_MODEL") or cfg.get("model", "nvidia/Cosmos3-Super-Reasoner"),
        "timeout_s": float(cfg.get("timeout_s", 40)),
        "max_tokens": int(cfg.get("max_tokens", 700)),
        "temperature": float(cfg.get("temperature", 0.2)),
        "image_max_side": int(cfg.get("image_max_side", 1024)),
        "annotate_image": bool(cfg.get("annotate_image", True)),
        "retries": int(cfg.get("retries", 1)),
        "mock": os.environ.get("COSMOS_MOCK", "") not in ("", "0") or bool(cfg.get("mock", False)),
        "send_image": str(os.environ.get("COSMOS_SEND_IMAGE") or cfg.get("send_image", "auto")).lower(),
    }


def model_name() -> str:
    """The model that will answer (for the on-screen label)."""
    s = _settings()
    return "offline mock" if s["mock"] else s["model"]


# ---------------------------------------------------------------- image prep
def annotate(frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Draw each object's box with "id:label" so the model can match list ids to the picture."""
    img = frame.copy()
    for o in objects:
        bbox = o.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        tag = f"{o['id']}:{o['label']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 255), -1)
        cv2.putText(img, tag, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def encode_image(frame: np.ndarray, max_side: int) -> str:
    """Downscale and JPEG-encode a BGR frame as a data URL."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not JPEG-encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_user_text(command: str, objects: list[dict]) -> str:
    listing = [
        {"id": o["id"], "label": o["label"], "x_cm": round(float(o["x"]), 1), "y_cm": round(float(o["y"]), 1)}
        for o in objects
    ]
    return (
        f'User request: "{command}"\n'
        f"Objects detected on the floor (boxes in the image are labelled id:label):\n"
        f"{json.dumps(listing)}\n"
        'Answer with only {"target_id": <int>, "reason": "<one sentence>"}.'
    )


# ---------------------------------------------------------------- parsing
_JSON_RE = re.compile(r"\{[^{}]*\"target_id\"[^{}]*\}", re.S)


def parse_answer(text: str, valid_ids: set[int]) -> dict | None:
    """Pull {"target_id", "reason"} out of the model's reply. Returns None if unusable."""
    if not text:
        return None
    if "</think>" in text:  # reasoner may wrap private reasoning in <think>...</think>
        text = text.rsplit("</think>", 1)[1]
    text = text.replace("```json", "").replace("```", "")
    for m in reversed(_JSON_RE.findall(text)):  # last JSON block is the final answer
        try:
            d = json.loads(m)
            tid = int(d["target_id"])
        except (ValueError, KeyError, TypeError):
            continue
        if tid != -1 and tid not in valid_ids:
            continue
        reason = " ".join(str(d.get("reason", "")).split()) or "No reason given."
        return {"target_id": tid, "reason": reason[:240]}
    return None


# ---------------------------------------------------------------- mock (offline)
_HINTS = {
    "write": ["pen", "pencil", "marker", "book"],
    "drink": ["cup", "bottle", "wine glass", "mug"],
    "thirst": ["bottle", "cup"],
    "cut": ["scissors", "knife"],
    "call": ["cell phone", "phone"],
    "phone": ["cell phone"],
    "eat": ["apple", "banana", "orange", "sandwich"],
    "hungry": ["banana", "apple", "orange"],
    "fruit": ["banana", "apple", "orange"],
    "read": ["book"],
    "ball": ["sports ball"],
    "play": ["sports ball", "teddy bear"],
    "time": ["clock"],
    "tv": ["remote"],
}


def _mock_choose(command: str, objects: list[dict]) -> dict:
    cmd = command.lower()
    for o in objects:  # direct mention wins
        if o["label"].lower() in cmd:
            return {"target_id": o["id"], "reason": f"[mock] You asked for a {o['label']} directly."}
    for key, labels in _HINTS.items():
        if key in cmd:
            for want in labels:
                for o in objects:
                    if o["label"].lower() == want:
                        return {"target_id": o["id"], "reason": f"[mock] A {want} matches '{key}'."}
    return {"target_id": -1, "reason": "[mock] Nothing in view matches that request."}


# ---------------------------------------------------------------- main entry
def choose_target(frame: np.ndarray, command: str, objects: list[dict]) -> dict:
    """Ask Cosmos which object to drive to. See module docstring for the contract."""
    s = _settings()
    t0 = time.time()
    last_call_info.clear()
    last_call_info.update({"model": s["model"], "mock": s["mock"], "ok": False})

    if not objects:
        last_call_info.update({"latency_s": 0.0, "ok": True})
        return {"target_id": -1, "reason": "I can't see any objects on the floor."}

    if s["mock"]:
        res = _mock_choose(command, objects)
        last_call_info.update({"latency_s": time.time() - t0, "ok": True})
        return res

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not set. Export it, or set COSMOS_MOCK=1 for offline sim tests.")

    from openai import OpenAI  # imported here so mock mode works without the package

    client = OpenAI(base_url=s["base_url"], api_key=api_key, timeout=s["timeout_s"], max_retries=0)
    text_part = {"type": "text", "text": build_user_text(command, objects)}
    use_image = s["send_image"] not in ("false", "0", "no") and s["model"] not in _TEXT_ONLY
    image_part = None
    if use_image:
        img = annotate(frame, objects) if s["annotate_image"] else frame
        image_part = {"type": "image_url", "image_url": {"url": encode_image(img, s["image_max_side"])}}

    def make_messages() -> list[dict]:
        content = [text_part, image_part] if use_image else [text_part]
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    messages = make_messages()
    valid = {int(o["id"]) for o in objects}
    raw = ""
    err: Exception | None = None
    attempts = s["retries"] + 1
    attempt = 0
    while attempt < attempts:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=s["model"],
                messages=messages,
                temperature=s["temperature"],
                max_tokens=s["max_tokens"],
            )
            msg = resp.choices[0].message
            raw = msg.content or ""
            last_call_info["text_only"] = not use_image
            last_call_info["usage"] = getattr(resp, "usage", None) and resp.usage.model_dump()
            parsed = parse_answer(raw, valid)
            if parsed is None and getattr(msg, "reasoning_content", None):
                parsed = parse_answer(msg.reasoning_content, valid)
            if parsed is not None:
                last_call_info.update({"latency_s": time.time() - t0, "ok": True, "raw": raw, "attempts": attempt})
                return parsed
            err = ValueError(f"could not parse a valid answer from: {raw[:300]!r}")
        except Exception as e:  # network, timeout, 4xx/5xx
            err = e
            status = getattr(e, "status_code", None)
            if use_image and s["send_image"] == "auto" and status in (400, 415, 422):
                # model probably doesn't take images: remember that and retry with text only
                _TEXT_ONLY.add(s["model"])
                use_image = False
                messages = make_messages()
                attempts += 1
    last_call_info.update({"latency_s": time.time() - t0, "raw": raw, "error": repr(err)})
    raise RuntimeError(f"Cosmos call failed: {err}")


# ---------------------------------------------------------------- CLI test
def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Send one photo + command to Cosmos and print its pick.")
    ap.add_argument("image")
    ap.add_argument("command")
    ap.add_argument("--labels", default="", help="comma-separated labels to offer as ids 0..n-1 (no YOLO needed)")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read image {args.image}")
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    objects = [{"id": i, "label": l, "conf": 1.0, "x": 0.0, "y": 0.0, "bbox": None} for i, l in enumerate(labels)]
    if not objects:
        sys.exit("give --labels so Cosmos has a list to choose from, e.g. --labels cup,scissors,pen")
    s = _settings()
    print(f"model={s['model']}  base_url={s['base_url']}  mock={s['mock']}  send_image={s['send_image']}")
    try:
        res = choose_target(frame, args.command, objects)
    finally:
        info = dict(last_call_info)
        raw = info.pop("raw", "")
        print("call info:", info)
        if raw:
            print("raw reply (last 600 chars):\n", raw[-600:])
    pick = next((o["label"] for o in objects if o["id"] == res["target_id"]), "nothing")
    print(f"\nPICK: id={res['target_id']} ({pick})\nREASON: {res['reason']}")


if __name__ == "__main__":
    _cli()
