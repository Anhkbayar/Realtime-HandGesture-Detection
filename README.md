# Hand Gesture-Based Human Computer Interaction

This project uses MediaPipe Hands and OpenCV to detect hand gestures from a webcam and control desktop interactions on the computer. The previous ESP32 serial workflow has been removed; no ESP32, COM port, relay, buzzer, or Arduino sketch is required.

## Features

- Live webcam overlay with hand landmarks.
- Gesture label, finger state, armed/disarmed state, and last action feedback.
- Safety clutch: hold an open palm for about one second to arm or disarm desktop control.
- Cursor movement, left click, right click, scrolling, volume control, and browser-style navigation gestures.

## Requirements

- Windows.
- Webcam. The app tries camera index `0` first, then scans nearby indices.
- Python 3.12 virtual environment recommended.
- MediaPipe is pinned to `0.10.21` because newer `0.10.33+` wheels expose only the Tasks API and no longer include `mp.solutions.hands`, which this webcam HCI demo uses.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is not available, install Python 3.12 first and repeat the setup.

If you previously installed unpinned dependencies and see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, reinstall the pinned dependencies:

```powershell
pip uninstall -y mediapipe opencv-python opencv-contrib-python numpy
pip install -r requirements.txt
```

## Run

```powershell
python gesture_control.py
```

If OpenCV chooses the wrong camera, set the camera index before running:

```powershell
$env:HAND_GESTURE_CAMERA_INDEX = "1"
python gesture_control.py
```

Press `Esc` while the webcam window is active to exit.

## Gestures

Start with the app disarmed. Hold an open palm steady for about one second to arm desktop control. Hold it again to disarm.

| Gesture | Action |
| --- | --- |
| Index finger only | Move cursor |
| Thumb + index pinch | Left click |
| Thumb + middle pinch | Right click |
| Index + middle raised | Scroll by moving hand up or down |
| Thumb up | Volume up |
| Thumb down | Volume down |
| Open palm swipe left | Navigate back |
| Open palm swipe right | Navigate forward |

PyAutoGUI fail-safe is enabled. Move the mouse to a screen corner to interrupt automation if needed.
