# rover-ai: context for AI coding assistants

3-person hackathon team, deadline Oct 30 2026, 10pm Singapore time (official: Oct 31, 1am SGT).
Overhead webcam + ESP32 rover. A typed command goes to NVIDIA Cosmos3-Super-Reasoner on
Nebius Token Factory, which picks an object from YOLO's list; the laptop then steers the rover to it.

## Interfaces (the team contract: do not change without telling the team)

- UDP to ESP32, port 4210: ASCII `L,R`, ints -255..255, about 15 Hz. ESP32 stops after 300 ms of silence.
- `vision/tracker.py`: `get_pose(frame) -> {"x": cm, "y": cm, "theta": rad} | None`
- `vision/detector.py`: `detect(frame) -> [{"id", "label", "conf", "x", "y", "bbox": [x1,y1,x2,y2]}]` (yolov8n, rover removed)
- `ai/cosmos.py`: `choose_target(frame, command, objects) -> {"target_id": int, "reason": str}` (-1 = nothing fits)
- `control/drive.py`: `step(pose, target_xy) -> (L, R, done)`; done within 8 cm. Gains in config.yaml.
- Floor frame (Track B): origin at corner marker 1, +x toward marker 2, +y toward marker 4,
  theta = atan2(dy, dx) in that frame. cm and radians.
- `vision/tracker.py` also exposes `px_to_floor(u, v)` and `floor_to_px(x, y)` (raw pixels).
  main.py uses them for click-to-drive, path drawing and working out the turn direction.
- `control.turn_sign: auto` derives the turn direction from the homography (assumes an
  unmirrored camera). Set 1 or -1 to override.
- config.yaml: Track B keys are flat (`arena_w_cm`, `camera_index`, ...); Track C keys are in
  sections (`rover`, `cosmos`, `control`, `mission`, `sim`).

## Rules

- Keep it simple. No reinforcement learning, training or Isaac Sim.
- Python 3.10+ on the laptop, Arduino framework on the ESP32.
- Cosmos is called once per command, never in the control loop.
- API key only from the `NEBIUS_API_KEY` environment variable. Never commit keys.
- Give complete runnable files plus the exact command to test each one.
- When something fails, ask for the error output before guessing.

## Owners

- Track A: firmware and rover hardware
- Track B: vision (tracker, detector)
- Track C: ai/cosmos.py, control/drive.py, sim/sim2d.py, main.py, README, Devpost

## Test commands

```bash
python -m tests.test_sim          # headless, no key needed
python main.py --mock             # sim with offline matcher
python main.py --mode real        # hardware
```
