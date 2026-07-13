# TouchlessOS

Control your Windows PC with hand gestures alone — no mouse, no keyboard.
A single webcam and [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand tracking
turn static poses into app launches and pinch gestures into real-time volume/brightness control.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

**Static gestures** — hold a pose for ~0.5s to trigger an action once:

| Gesture | Action |
|---|---|
| ✋ Open palm | Launch Chrome |
| ✊ Fist | Launch VS Code |
| ✌️ Peace sign | Open File Explorer |
| 👍 Thumbs up | Launch Calculator |
| 🖼️ Square frame (both hands, thumbs + index fingers touching) | Open ChatGPT in Chrome |

**Continuous gestures** — pinch thumb and index finger, other fingers curled:

| Hand | Controls |
|---|---|
| Left hand pinch | System volume — spread fingers to raise, pinch closed to lower |
| Right hand pinch | Screen brightness — spread fingers to raise, pinch closed to lower |

Both hands can be used at the same time, independently.

## How it works

1. **OpenCV** captures webcam frames and handles the live preview window.
2. **MediaPipe Hands** locates each hand in the frame and returns 21 3D landmark
   points per hand (fingertips, knuckles, wrist) plus left/right handedness.
3. Custom geometry logic (`gestures.py`) turns those landmarks into gesture
   labels — e.g. a finger is "up" if its tip sits above its middle knuckle,
   a "pinch" is measured as the thumb-to-index distance normalized by hand
   size so it works at any distance from the camera.
4. A gesture held steady for a set number of frames triggers an action
   (`actions.py`) or, for pinches, continuously drives system volume/brightness
   (`system_control.py`) via `pycaw` and `screen_brightness_control`.
5. A short flicker-tolerance window absorbs brief tracking dropouts so a
   held gesture doesn't reset just because MediaPipe missed a single frame.

## Demo

*(Add a GIF or short screen recording here showing a gesture triggering an app launch and a pinch adjusting volume — this is the single highest-impact addition for LinkedIn/portfolio viewers.)*

## Getting started

### Requirements

- Windows 10/11
- Python 3.9+
- A webcam

### Installation

```bash
git clone https://github.com/<your-username>/touchless-os.git
cd touchless-os
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

A window opens showing your webcam feed with hand landmarks overlaid and the
currently recognized gesture. Press **q** to quit.

## Project structure

```
main.py             # Webcam loop, hold/debounce logic, wires gestures to actions
gestures.py          # Landmark geometry -> gesture classification
actions.py           # Gesture -> app-launch command mapping
system_control.py    # Windows volume (pycaw) and brightness (screen_brightness_control)
requirements.txt
```

## Configuration

Gesture-to-app mappings live in `GESTURE_TO_APP` in `actions.py` — add or
change entries to launch whatever you like. Key tunables in `main.py`:

| Constant | Purpose |
|---|---|
| `HOLD_FRAMES` | How many consecutive frames a static gesture must be held before it triggers |
| `COOLDOWN_SECONDS` | Minimum time between two triggered actions |
| `MISS_TOLERANCE_FRAMES` | How many dropped-tracking frames are tolerated before a hold resets |
| `PINCH_SMOOTHING` | Smoothing factor for pinch-based volume/brightness (lower = smoother, slower to respond) |

## Limitations

- Windows-only (app launching and system control both use Windows-specific APIs).
- Static gestures are rule-based on finger up/down geometry, not a trained
  classifier — works reliably for the built-in gesture set but isn't a
  general-purpose sign language recognizer.
- Requires reasonable, consistent lighting for MediaPipe to track reliably.

## License

MIT — see [LICENSE](LICENSE).
