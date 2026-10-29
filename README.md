# Fetch Rover: an overhead-camera AI rover

Say what you need, and a small rover drives to the right object. An overhead webcam watches the floor, YOLO finds the objects, and **NVIDIA Cosmos3-Super-Reasoner running on Nebius Token Factory** decides which object answers your request and explains why. A fast control loop on the laptop then steers an ESP32 rover to it.

Built for the Nebius x NVIDIA Global AI Hackathon (Physical AI track).

## Where Nebius and Cosmos are used

| What | Where |
| --- | --- |
| Nebius Token Factory client (OpenAI-compatible API, key from `NEBIUS_API_KEY`) | [`ai/cosmos.py` line 186](ai/cosmos.py#L186) |
| Call to `nvidia/Cosmos3-Super-Reasoner` with the camera frame, the object list and the user's command | [`ai/cosmos.py` line 204](ai/cosmos.py#L204) |
| Endpoint and model settings | [`config.yaml`](config.yaml) → `cosmos:` |
| Cosmos's choice and reasoning shown on screen | [`main.py`](main.py) → `poll_cosmos()` and `draw_panel()` |

Cosmos is called **once per command**, never inside the steering loop. It picks an id from YOLO's list rather than returning pixel coordinates, which is far more reliable.

## How it works

Two loops run on one laptop:

1. **Slow AI loop (once per command).** You type a command. YOLO lists the objects with their floor positions. `ai/cosmos.py` sends one annotated frame, the list and the command to Cosmos3-Super-Reasoner on Nebius. Cosmos returns `{"target_id", "reason"}` and the reason appears on screen.
2. **Fast control loop (about 15 Hz).** The tracker reads the rover's ArUco marker for its position and heading in floor cm. `control/drive.py` turns the rover in place toward the target, then drives forward with proportional steering. It sends `L,R` motor speeds over WiFi (UDP) to the ESP32 and stops within 8 cm.

```
webcam ─► tracker (ArUco pose) ──────────────┐
      └─► detector (YOLO objects) ─┐         ▼
typed command ─► Cosmos on Nebius ─┴► target ─► controller ─► UDP "L,R" ─► ESP32 ─► motors
                 (once per command)            (15 Hz)                         │
      ▲───────────────────── camera sees the rover move ◄──────────────────────┘
```

## Repo layout

```
firmware/rover_esp32/rover_esp32.ino   ESP32 UDP motor firmware with 300 ms failsafe
vision/tracker.py                      get_pose(frame) -> {"x","y","theta"} | None; px_to_floor / floor_to_px
vision/view.py                         live debug viewer for the camera and markers
vision/detector.py                     detect(frame) -> [{"id","label","conf","x","y","bbox"}]
ai/cosmos.py                           choose_target(frame, command, objects) -> {"target_id","reason"}
control/drive.py                       step(pose, target_xy) -> (L, R, done), plus UdpSender
sim/sim2d.py                           2D simulator with the same interfaces (no hardware needed)
main.py                                main loop and on-screen overlay
config.yaml                            IPs, ports, gains, arena size, Cosmos endpoint
tests/test_sim.py                      headless tests
```

Floor frame: origin at corner marker 1, +x toward marker 2, +y toward marker 4, theta = atan2(dy, dx), all in cm and radians. `control.turn_sign: auto` works out from the homography which way the rover must turn, so the corner markers can be laid out either way round.

## Hardware

| Part | Notes |
| --- | --- |
| Acrylic 2WD chassis kit | 2 TT motors and a caster |
| ESP32 DevKit | Joins a phone hotspot, listens on UDP port 4210 |
| TB6612FNG motor driver | Wiring below |
| Battery + 5V buck converter | 2× 18650 or AA pack; common ground with everything |
| 1080p wide-angle webcam | About 2 m up, pointing straight down |
| ArUco markers (DICT_4X4_50) | ID 0 on the rover (8–10 cm), IDs 1–4 at the arena corners |

| TB6612FNG | ESP32 | Notes |
| --- | --- | --- |
| PWMA | GPIO 25 | Left motor speed |
| AIN1 / AIN2 | GPIO 26 / 27 | Left motor direction |
| PWMB | GPIO 14 | Right motor speed |
| BIN1 / BIN2 | GPIO 32 / 33 | Right motor direction |
| STBY | GPIO 4 (or 3.3V) | High = enabled |
| VCC | 3.3V | Logic power |
| VM | Battery + after the switch | Motor power |
| GND | Common ground | Battery, buck, ESP32 and driver |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m vision.view                                   # check camera, corner markers and rover pose first
export NEBIUS_API_KEY=v1.xxxxx                          # Windows: set NEBIUS_API_KEY=v1.xxxxx
```

If the organisers gave you a dedicated Cosmos endpoint, set its URL in `config.yaml` → `cosmos.base_url` (or `export NEBIUS_BASE_URL=...`).

## Run

```bash
python -m tests.test_sim                     # headless tests, no key or hardware needed
python main.py --mock                        # simulator with an offline keyword matcher
python main.py                               # simulator with real Cosmos calls
python main.py --mode real                   # webcam + rover
python main.py --mode real --record demo.mp4 # also save the overlay for the video
```

Test one photo against Cosmos:

```bash
python -m ai.cosmos photo.jpg "I need to cut some paper" --labels cup,scissors,cell\ phone
```

Hardware smoke test (forward, back, left, right for a second each):

```bash
python -m control.drive --spin
```

Fake ESP32 for testing UDP code without hardware: `python -m sim.sim2d --udp`, then send to `127.0.0.1:4210`.

In the window, type a command and press Enter. Click the floor to drive to a point. **Esc** stops the rover. `/goto X Y` drives to a floor point in cm. `/quit` exits.

## Tuning on the real rover

| Symptom | Fix in `config.yaml` → `control` |
| --- | --- |
| Turns away from the target | Set `turn_sign` to `1` or `-1` instead of `auto` (needed if the camera image is mirrored) |
| Wheels hum but don't move at low speed | Raise `min_pwm` (it must sit above the motors' deadband) |
| Overshoots or wobbles | Lower `forward_pwm` first, then `k_heading` |
| Spins past the heading | Lower `turn_pwm` |
| Bumps into the object | Raise `mission.stop_short_cm` |

## License

MIT, see [LICENSE](LICENSE).
