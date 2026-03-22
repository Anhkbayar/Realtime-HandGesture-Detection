import cv2
import mediapipe as mp
import serial
import time

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

TARGET = "1,0,1,0,0"
countdown_active = False
countdown_start = 0
countdown_value = 3

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_data = ""

def get_finger_state(landmarks):
    fingers = []

    # Thumb (x comparison)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (y comparison)
    tips = [8, 12, 16, 20]

    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # [thumb, index, middle, ring, pinky]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    data_str = "0,0,0,0,0"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            fingers = get_finger_state(handLms.landmark)

            data_str = ",".join(map(str, fingers))
            
            if data_str == TARGET and not countdown_active:
                countdown_active = True
                countdown_start = time.time()
                countdown_value = 3

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Send only if changed
    if data_str != prev_data:
        ser.write((data_str + "\n").encode())
        prev_data = data_str

    cv2.putText(frame, f"{data_str}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Countdown logic
    if data_str == TARGET:
        if not countdown_active:
            countdown_active = True
            countdown_start = time.time()

        elapsed = int(time.time() - countdown_start)

        if elapsed < 3:
            countdown_value = 3 - elapsed
            cv2.putText(frame, f"Starting in {countdown_value}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "KABOOM!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    else:
        #Gesture gone reset
        countdown_active = False
        countdown_start = 0

    cv2.imshow("Finger Mapping", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()