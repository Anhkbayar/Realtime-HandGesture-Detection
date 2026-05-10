import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


CAMERA_INDEX = 0
CAMERA_INDEX_ENV = "HAND_GESTURE_CAMERA_INDEX"
CAMERA_SCAN_LIMIT = 10
CAMERA_RESOLUTIONS = ((1280, 720), (960, 540), (640, 480), None)
CAMERA_WARMUP_READS = 20
CAMERA_WARMUP_DELAY_SECONDS = 0.05
BLACK_FRAME_MEAN_THRESHOLD = 5.0
BLACK_FRAME_STD_THRESHOLD = 3.0
MAX_CONSECUTIVE_CAMERA_READ_FAILURES = 30
WINDOW_NAME = "Hand Gesture HCI"

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
MODEL_COMPLEXITY = 0

PINCH_DISTANCE = 0.045
PINCH_PREP_DISTANCE = 0.075
CLICK_COOLDOWN_SECONDS = 0.55
KEY_COOLDOWN_SECONDS = 0.65
CLUTCH_HOLD_SECONDS = 1.0
CLUTCH_RELEASE_SECONDS = 0.45
FINGER_EXTENSION_MARGIN = 0.025
PEACE_DIRECTION_MARGIN = 0.035
THUMB_DIRECTION_MARGIN = 0.07
SCROLL_STEP = 14
SCROLL_INTERVAL_SECONDS = 0.03
CURSOR_SMOOTHING = 0.25

OVERLAY_GREEN = (30, 220, 30)
OVERLAY_RED = (40, 40, 230)
OVERLAY_BLUE = (230, 150, 40)
OVERLAY_WHITE = (245, 245, 245)
OVERLAY_BLACK = (0, 0, 0)

CAMERA_BACKENDS = (
    ("MSMF", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
    ("Default", cv2.CAP_ANY),
)


@dataclass
class RuntimeState:
    armed: bool = False
    clutch_started_at: Optional[float] = None
    clutch_locked_until_release: bool = False
    open_palm_released_at: Optional[float] = None
    last_scroll_at: float = 0.0
    cursor_position: Optional[Tuple[float, float]] = None
    last_action: str = "None"
    last_left_click_at: float = 0.0
    last_right_click_at: float = 0.0
    last_key_at: float = 0.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def landmark_distance(landmarks: Sequence, first: int, second: int) -> float:
    first_point = landmarks[first]
    second_point = landmarks[second]
    return float(
        np.hypot(first_point.x - second_point.x, first_point.y - second_point.y)
    )


def get_finger_state(landmarks: Sequence) -> Tuple[int, int, int, int, int]:
    """Return thumb, index, middle, ring, pinky as 1 when the finger is extended."""
    thumb_extended = landmark_distance(landmarks, 4, 17) > landmark_distance(
        landmarks, 3, 17
    )

    fingers = [1 if thumb_extended else 0]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

    return tuple(fingers)


def finger_extended_from_wrist(landmarks: Sequence, tip: int, pip: int) -> bool:
    return landmark_distance(landmarks, 0, tip) > (
        landmark_distance(landmarks, 0, pip) + FINGER_EXTENSION_MARGIN
    )


def get_peace_sign_direction(landmarks: Sequence) -> Optional[str]:
    index_extended = finger_extended_from_wrist(landmarks, 8, 6)
    middle_extended = finger_extended_from_wrist(landmarks, 12, 10)
    ring_folded = not finger_extended_from_wrist(landmarks, 16, 14)
    pinky_folded = not finger_extended_from_wrist(landmarks, 20, 18)

    if not (index_extended and middle_extended and ring_folded and pinky_folded):
        return None

    tip_y = (landmarks[8].y + landmarks[12].y) / 2
    pip_y = (landmarks[6].y + landmarks[10].y) / 2

    if tip_y < pip_y - PEACE_DIRECTION_MARGIN:
        return "up"

    if tip_y > pip_y + PEACE_DIRECTION_MARGIN:
        return "down"

    return None


def recognize_gesture(landmarks: Sequence, fingers: Tuple[int, int, int, int, int]) -> str:
    thumb_index_distance = landmark_distance(landmarks, 4, 8)
    thumb_middle_distance = landmark_distance(landmarks, 4, 12)
    thumb_index_pinched = thumb_index_distance < PINCH_DISTANCE
    thumb_middle_pinched = thumb_middle_distance < PINCH_DISTANCE
    click_prep = (
        thumb_index_distance < PINCH_PREP_DISTANCE
        or thumb_middle_distance < PINCH_PREP_DISTANCE
    )

    if all(fingers):
        return "Open Palm"

    if thumb_index_pinched:
        return "Left Click Pinch"

    if thumb_middle_pinched:
        return "Right Click Pinch"

    if click_prep:
        return "Click Prep"

    peace_direction = get_peace_sign_direction(landmarks)
    if peace_direction == "up":
        return "Scroll Up"

    if peace_direction == "down":
        return "Scroll Down"

    if fingers[1] == 1 and fingers[2:] == (0, 0, 0):
        return "Cursor Move"

    if fingers[0] == 1 and fingers[1:] == (0, 0, 0, 0):
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        thumb_dx = thumb_tip.x - wrist.x
        thumb_dy = thumb_tip.y - wrist.y

        if abs(thumb_dx) > max(abs(thumb_dy), THUMB_DIRECTION_MARGIN):
            if thumb_dx < 0:
                return "Navigate Back"

            return "Navigate Forward"

        if thumb_dy < -THUMB_DIRECTION_MARGIN:
            return "Volume Up"

        if thumb_dy > THUMB_DIRECTION_MARGIN:
            return "Volume Down"

    return "Unknown"


def can_fire(last_fired_at: float, cooldown: float, now: float) -> bool:
    return now - last_fired_at >= cooldown


def set_last_action(state: RuntimeState, action: str) -> None:
    state.last_action = action


def handle_clutch(state: RuntimeState, gesture: str, now: float) -> None:
    if gesture != "Open Palm":
        if state.clutch_locked_until_release:
            if state.open_palm_released_at is None:
                state.open_palm_released_at = now
            elif now - state.open_palm_released_at >= CLUTCH_RELEASE_SECONDS:
                state.clutch_locked_until_release = False

        state.clutch_started_at = None
        return

    state.open_palm_released_at = None

    if state.clutch_locked_until_release:
        return

    if state.clutch_started_at is None:
        state.clutch_started_at = now
        return

    if now - state.clutch_started_at >= CLUTCH_HOLD_SECONDS:
        state.armed = not state.armed
        state.clutch_locked_until_release = True
        state.clutch_started_at = None
        set_last_action(state, "Control armed" if state.armed else "Control disarmed")


def move_cursor(state: RuntimeState, landmarks: Sequence, screen_size: Tuple[int, int]) -> None:
    screen_width, screen_height = screen_size
    index_tip = landmarks[8]

    target_x = clamp(index_tip.x * screen_width, 0, screen_width - 1)
    target_y = clamp(index_tip.y * screen_height, 0, screen_height - 1)

    if state.cursor_position is None:
        smoothed_x, smoothed_y = target_x, target_y
    else:
        previous_x, previous_y = state.cursor_position
        smoothed_x = previous_x + (target_x - previous_x) * CURSOR_SMOOTHING
        smoothed_y = previous_y + (target_y - previous_y) * CURSOR_SMOOTHING

    state.cursor_position = (smoothed_x, smoothed_y)
    pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0)
    set_last_action(state, "Cursor move")


def scroll_from_hand(state: RuntimeState, gesture: str, now: float) -> None:
    if now - state.last_scroll_at < SCROLL_INTERVAL_SECONDS:
        return

    scroll_amount = SCROLL_STEP if gesture == "Scroll Up" else -SCROLL_STEP
    pyautogui.scroll(scroll_amount)
    state.last_scroll_at = now
    set_last_action(state, gesture)


def press_hotkey(keys: Sequence[str]) -> None:
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)


def execute_action(
    state: RuntimeState,
    gesture: str,
    landmarks: Sequence,
    screen_size: Tuple[int, int],
    now: float,
) -> None:
    if not state.armed:
        state.last_scroll_at = 0.0
        return

    cursor_stable_gestures = (
        "Cursor Move",
        "Click Prep",
        "Left Click Pinch",
        "Right Click Pinch",
    )
    if gesture not in cursor_stable_gestures:
        state.cursor_position = None

    if gesture not in ("Scroll Up", "Scroll Down"):
        state.last_scroll_at = 0.0

    if gesture == "Cursor Move":
        move_cursor(state, landmarks, screen_size)
        return

    if gesture == "Click Prep":
        set_last_action(state, "Hold cursor for click")
        return

    if gesture == "Left Click Pinch" and can_fire(
        state.last_left_click_at, CLICK_COOLDOWN_SECONDS, now
    ):
        pyautogui.click(button="left")
        state.last_left_click_at = now
        set_last_action(state, "Left click")
        return

    if gesture == "Right Click Pinch" and can_fire(
        state.last_right_click_at, CLICK_COOLDOWN_SECONDS, now
    ):
        pyautogui.click(button="right")
        state.last_right_click_at = now
        set_last_action(state, "Right click")
        return

    if gesture in ("Scroll Up", "Scroll Down"):
        scroll_from_hand(state, gesture, now)
        return

    if gesture == "Volume Up" and can_fire(state.last_key_at, KEY_COOLDOWN_SECONDS, now):
        pyautogui.press("volumeup")
        state.last_key_at = now
        set_last_action(state, "Volume up")
        return

    if gesture == "Volume Down" and can_fire(state.last_key_at, KEY_COOLDOWN_SECONDS, now):
        pyautogui.press("volumedown")
        state.last_key_at = now
        set_last_action(state, "Volume down")
        return

    if gesture == "Navigate Back" and can_fire(
        state.last_key_at, KEY_COOLDOWN_SECONDS, now
    ):
        press_hotkey(("alt", "left"))
        state.last_key_at = now
        set_last_action(state, "Navigate back")
        return

    if gesture == "Navigate Forward" and can_fire(
        state.last_key_at, KEY_COOLDOWN_SECONDS, now
    ):
        press_hotkey(("alt", "right"))
        state.last_key_at = now
        set_last_action(state, "Navigate forward")


def draw_overlay(
    frame,
    state: RuntimeState,
    detected: bool,
    gesture: str,
    fingers: Tuple[int, int, int, int, int],
) -> None:
    status_text = "ARMED" if state.armed else "DISARMED"
    status_color = OVERLAY_GREEN if state.armed else OVERLAY_RED

    lines = [
        ("Hand detected" if detected else "No hand detected", OVERLAY_WHITE),
        (f"State: {status_text}", status_color),
        (f"Gesture: {gesture}", OVERLAY_BLUE),
        (f"Fingers: {','.join(map(str, fingers))}", OVERLAY_WHITE),
        (f"Last action: {state.last_action}", OVERLAY_WHITE),
        ("Hold open palm to arm/disarm | Esc to quit", OVERLAY_WHITE),
    ]

    y = 32
    for text, color in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30


def show_startup_frame(cap) -> None:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        return

    frame = cv2.flip(frame, 1)
    cv2.putText(
        frame,
        "Camera ready. Starting hand tracking...",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        OVERLAY_BLACK,
        4,
    )
    cv2.putText(
        frame,
        "Camera ready. Starting hand tracking...",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        OVERLAY_WHITE,
        2,
    )
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)


def get_preferred_camera_index() -> int:
    raw_index = os.getenv(CAMERA_INDEX_ENV)
    if raw_index is None:
        return CAMERA_INDEX

    try:
        return int(raw_index)
    except ValueError as exc:
        raise RuntimeError(
            f"{CAMERA_INDEX_ENV} must be an integer camera index, got {raw_index!r}."
        ) from exc


def camera_index_candidates(preferred_index: int) -> Tuple[int, ...]:
    candidates = [preferred_index]
    for index in range(CAMERA_SCAN_LIMIT + 1):
        if index not in candidates:
            candidates.append(index)
    return tuple(candidates)


def frame_has_visible_image(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return not (
        float(gray.mean()) < BLACK_FRAME_MEAN_THRESHOLD
        and float(gray.std()) < BLACK_FRAME_STD_THRESHOLD
    )


def read_valid_frame(cap) -> bool:
    for _ in range(CAMERA_WARMUP_READS):
        ret, frame = cap.read()
        if (
            ret
            and frame is not None
            and frame.size > 0
            and frame_has_visible_image(frame)
        ):
            return True
        time.sleep(CAMERA_WARMUP_DELAY_SECONDS)
    return False


def request_camera_resolution(cap, resolution: Optional[Tuple[int, int]]) -> None:
    if resolution is None:
        return

    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def describe_camera_resolution(cap) -> str:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f"{width}x{height}"


def describe_requested_resolution(resolution: Optional[Tuple[int, int]]) -> str:
    if resolution is None:
        return "default resolution"

    width, height = resolution
    return f"{width}x{height}"


def open_camera():
    attempts = []

    for index in camera_index_candidates(get_preferred_camera_index()):
        for backend_name, backend in CAMERA_BACKENDS:
            for resolution in CAMERA_RESOLUTIONS:
                cap = cv2.VideoCapture(index, backend)
                requested_resolution = describe_requested_resolution(resolution)
                if not cap.isOpened():
                    attempts.append(
                        f"index {index} via {backend_name} at "
                        f"{requested_resolution}: open failed"
                    )
                    cap.release()
                    break

                request_camera_resolution(cap, resolution)

                if read_valid_frame(cap):
                    print(
                        f"Opened camera index {index} via {backend_name} "
                        f"at {describe_camera_resolution(cap)}.",
                        flush=True,
                    )
                    return cap

                attempts.append(
                    f"index {index} via {backend_name} at "
                    f"{requested_resolution}: opened but no visible frames"
                )
                cap.release()

    details = "\n".join(f"- {attempt}" for attempt in attempts)
    raise RuntimeError(
        "Unable to open a webcam and read frames.\n"
        "Close other apps using the camera, remove any camera cover, check Windows "
        f"camera privacy settings, or set {CAMERA_INDEX_ENV} to the correct camera index.\n"
        f"Attempts:\n{details}"
    )


def main() -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "This app requires the MediaPipe legacy solutions API. "
            "Run `pip install -r requirements.txt` to install mediapipe==0.10.21."
        )

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    screen_size = pyautogui.size()
    state = RuntimeState()

    cap = open_camera()
    show_startup_frame(cap)
    print("Starting hand tracking. Press Esc in the camera window to quit.", flush=True)

    try:
        with mp_hands.Hands(
            model_complexity=MODEL_COMPLEXITY,
            max_num_hands=1,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        ) as hands:
            failed_reads = 0

            while True:
                ret, frame = cap.read()
                if (
                    not ret
                    or frame is None
                    or frame.size == 0
                    or not frame_has_visible_image(frame)
                ):
                    failed_reads += 1
                    if failed_reads >= MAX_CONSECUTIVE_CAMERA_READ_FAILURES:
                        raise RuntimeError(
                            "Camera opened, but it stopped returning visible frames. "
                            "Remove any camera cover, close other camera apps, or try "
                            f"a different {CAMERA_INDEX_ENV} value."
                        )
                    time.sleep(CAMERA_WARMUP_DELAY_SECONDS)
                    continue

                failed_reads = 0
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                detected = False
                gesture = "None"
                fingers = (0, 0, 0, 0, 0)
                now = time.time()

                if result.multi_hand_landmarks:
                    detected = True
                    hand_landmarks = result.multi_hand_landmarks[0]
                    landmarks = hand_landmarks.landmark
                    fingers = get_finger_state(landmarks)
                    gesture = recognize_gesture(landmarks, fingers)

                    handle_clutch(state, gesture, now)
                    execute_action(state, gesture, landmarks, screen_size, now)

                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                else:
                    handle_clutch(state, gesture, now)
                    state.cursor_position = None
                    state.last_scroll_at = 0.0

                draw_overlay(frame, state, detected, gesture, fingers)
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
