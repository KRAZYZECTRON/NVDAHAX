"""Headless checks for Track C. Run from the repo root:  python -m tests.test_sim

1. Controller drives the simulated rover to 20 random points within done_dist_cm.
2. Cosmos answer parsing handles think tags, code fences and bad ids.
3. The full command -> choose_target -> drive pipeline works with the mock Cosmos.
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import cosmos
from common import load_config
from control.drive import Controller
from sim.sim2d import SimSender, SimWorld

DT = 1 / 30  # camera-rate physics step


def drive_to(world: SimWorld, target, ctrl: Controller, timeout_s: float = 40.0):
    sender = SimSender(world)
    ctrl.reset()
    t0 = world.t
    while world.t - t0 < timeout_s:
        pose = world.get_pose()
        left, right, done = ctrl.step(pose, target)
        if done:
            sender.stop()
            return True, world.t - t0, math.hypot(world.x - target[0], world.y - target[1])
        sender.send(left, right)
        world.update(DT)
    return False, timeout_s, math.hypot(world.x - target[0], world.y - target[1])


def test_controller(n: int = 20) -> None:
    cfg = load_config()
    rng = random.Random(1)
    world = SimWorld(cfg, seed=3)
    ctrl = Controller(cfg)
    ok_count, times, errs = 0, [], []
    for _ in range(n):
        tgt = (rng.uniform(15, world.W - 15), rng.uniform(15, world.H - 15))
        ok, t, err = drive_to(world, tgt, ctrl)
        ok_count += ok
        times.append(t)
        errs.append(err)
    print(f"controller: {ok_count}/{n} reached | mean time {sum(times)/n:.1f}s | "
          f"max final error {max(errs):.1f} cm (true pose, done at {cfg['control']['done_dist_cm']} cm)")
    assert ok_count == n, "controller failed to reach some targets"


def test_parsing() -> None:
    ids = {0, 1, 2}
    cases = [
        ('{"target_id": 1, "reason": "Scissors cut paper."}', 1),
        ('<think>maybe the cup {"target_id": 0}... no</think>\n{"target_id": 2, "reason": "x"}', 2),
        ('```json\n{"target_id": 0, "reason": "a cup holds water"}\n```', 0),
        ('{"target_id": 9, "reason": "not in list"}', None),
        ('{"target_id": -1, "reason": "nothing fits"}', -1),
        ("I would pick the cup.", None),
    ]
    for text, want in cases:
        got = cosmos.parse_answer(text, ids)
        got_id = None if got is None else got["target_id"]
        assert got_id == want, (text, got)
    print(f"parsing: {len(cases)}/{len(cases)} cases ok")


def test_pipeline() -> None:
    os.environ["COSMOS_MOCK"] = "1"
    world = SimWorld(objects=[("cup", 40, 110), ("scissors", 150, 120), ("cell phone", 100, 80)])
    frame = world.render()
    objs = world.detect(frame)
    res = cosmos.choose_target(frame, "I need to cut some paper", objs)
    assert set(res) == {"target_id", "reason"}
    tgt = next(o for o in objs if o["id"] == res["target_id"])
    assert tgt["label"] == "scissors", res
    ok, t, err = drive_to(world, (tgt["x"], tgt["y"]), Controller())
    assert ok, "pipeline drive failed"
    print(f"pipeline: picked {tgt['label']} ({res['reason']}) and reached it in {t:.1f}s")


if __name__ == "__main__":
    test_parsing()
    test_controller()
    test_pipeline()
    print("ALL OK")
