import cv2
import mediapipe as mp
import os
import time

GESTURES = ["fist", "one_finger", "peace", "three_up", "open_palm", "pinch"]
DATA_DIR = "gesture_data"

for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

print("Folders ban gaye:", os.listdir(DATA_DIR))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

current_gesture_idx = 0
collecting = False
count = 0
TARGET_PER_GESTURE = 300

print("\n=== DATA COLLECTION SHURU ===")
print(f"Pehla gesture: '{GESTURES[0]}'")
print("'s' dabao = collecting shuru karo")
print("'n' dabao = next gesture skip karo")
print("'q' dabao = quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        lm_list = results.multi_hand_landmarks[0].landmark
        x_coords = [int(l.x * w) for l in lm_list]
        y_coords = [int(l.y * h) for l in lm_list]

        #hand cropping ke liye y and x ke max min coordinates
        x_min = max(0, min(x_coords) - 30)
        y_min = max(0, min(y_coords) - 30)
        x_max = min(w, max(x_coords) + 30)
        y_max = min(h, max(y_coords) + 30)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        hand_crop = frame[y_min:y_max, x_min:x_max]

        if collecting and current_gesture_idx < len(GESTURES):
            gesture_name = GESTURES[current_gesture_idx]
            save_path = os.path.join(DATA_DIR, gesture_name, f"{count}.jpg")
            if hand_crop.size > 0:
                resized = cv2.resize(hand_crop, (64, 64))
                cv2.imwrite(save_path, resized)
                count += 1

            if count >= TARGET_PER_GESTURE:
                print(f"✓ '{gesture_name}' done! {count} images save hui.")
                collecting = False
                count = 0
                current_gesture_idx += 1
                if current_gesture_idx < len(GESTURES):
                    print(f"Ab karo: '{GESTURES[current_gesture_idx]}'")
                    print("'s' dabao shuru karne ke liye")
                else:
                    print("\n Sabhi gestures collect ho gaye!")
                    print("Ab train_model.py chalao.")

    # Screen display
    if current_gesture_idx < len(GESTURES):
        gesture_label = GESTURES[current_gesture_idx]
    else:
        gesture_label = "COMPLETE!"

    color = (0, 255, 0) if collecting else (0, 165, 255)
    status = f"{'SAVING...' if collecting else 'READY'}: {gesture_label}"

    cv2.rectangle(frame, (0, 0), (1280, 80), (30, 30, 30), -1)
    cv2.putText(frame, status, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(frame, f"Count: {count} / {TARGET_PER_GESTURE}", (900, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if not hand_detected:
        cv2.putText(frame, "HAATH DIKHA! Camera ke saamne lao",
                    (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Data Collection - 's' shuru, 'n' skip, 'q' quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if hand_detected and current_gesture_idx < len(GESTURES):
            collecting = True
            count = 0
            print(f"Collecting shuru: '{GESTURES[current_gesture_idx]}'")
        elif not hand_detected:
            print("Pehle haath camera ke saamne lao!")
    elif key == ord('n'):
        collecting = False
        count = 0
        if current_gesture_idx + 1 < len(GESTURES):
            current_gesture_idx += 1
            print(f"Skip → Ab karo: '{GESTURES[current_gesture_idx]}'")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nCollection khatam! gesture_data/ folder check karo.")