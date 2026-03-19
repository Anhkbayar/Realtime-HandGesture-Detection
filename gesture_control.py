import cv2
import mediapipe as mp
import serial
import time

#CHANGE THIS to your COM port
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# Initialize serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Camera setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def is_hand_open(landmarks):
    tips = [8, 12, 16, 20]
    count = 0

    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    return count >= 3

prev_state = None  # prevent spam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture = "NONE"
    state = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            if is_hand_open(handLms.landmark):
                gesture = "OPEN"
                state = '1'
            else:
                gesture = "FIST"
                state = '0'

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Send only if state changed
    if state is not None and state != prev_state:
        ser.write(state.encode())
        prev_state = state

    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()
