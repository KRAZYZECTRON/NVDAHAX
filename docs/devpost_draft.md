# Devpost draft: Fetch Rover

> Draft. Replace every [bracket] with real numbers and details after testing.

## Inspiration
Asking for "something to cut this with" is easy for a person and hard for a robot. Most home robots need you to name an exact object or point at a spot. We wanted a robot that understands what you need and goes to the right thing, using a cheap camera and a $15 rover instead of expensive onboard sensors.

## What it does
You type a request such as "I need to cut some paper". An overhead camera sees the floor. NVIDIA Cosmos3-Super-Reasoner, running on Nebius Token Factory, looks at the scene and the list of detected objects, picks the one that fits, and explains its choice on screen. The rover then drives to that object and stops in front of it.

## How we built it
- **Perception:** a 1080p webcam about 2 m above the floor. ArUco markers give the rover's position and heading in centimetres; YOLOv8n finds the objects.
- **Reasoning:** one call per command to `nvidia/Cosmos3-Super-Reasoner` through Nebius Token Factory's OpenAI-compatible API. Cosmos receives the frame with numbered boxes, the object list and the request, and returns a JSON choice with a one-sentence reason.
- **Control:** a 15 Hz turn-then-drive controller on the laptop sends `L,R` motor speeds over WiFi UDP to an ESP32 with a TB6612FNG driver. A 300 ms failsafe stops the rover if packets stop.
- **Simulator:** a 2D simulator with the same interfaces let us build and tune the AI and control code before the hardware was ready.

## Challenges we ran into
- [e.g. Cosmos latency of N s per call: we call it once per command and show "thinking" on screen]
- [e.g. Marker parallax near the arena edges]
- [e.g. Motor deadband with weak batteries]
- Letting Cosmos choose from YOLO's numbered list instead of returning pixel coordinates made its answers reliable.

## Accomplishments that we're proud of
- [Rover reaches the chosen object within 8 cm in N of M runs]
- A full physical AI loop built in about 33 hours by three people.

## What we learned
- [Fill in]

## What's next
- Pick-up with a small gripper, so it fetches instead of just going to the object.
- Multi-step requests ("bring the cup, then the phone").
- Ceiling cameras covering a whole room.

## Built with
Python, OpenCV, YOLOv8, NVIDIA Cosmos3-Super-Reasoner, Nebius Token Factory, ESP32, Arduino

## Feedback on Nebius and NVIDIA tools
- **Nebius Token Factory:** [setup time, OpenAI-compatible API ease of use, latency per call, credits, anything confusing]
- **Cosmos3-Super-Reasoner:** [answer quality on top-down scenes, JSON reliability, think-tag handling, latency, what we'd want next]
