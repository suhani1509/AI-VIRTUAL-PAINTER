
import math
import cv2
import numpy as np
import time
import requests
import os
import threading
import HandTrackingModule as htm

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

API_URL = "http://localhost:8000/predict"
CONFIDENCE_THRESHOLD = 0.75
BRUSH_THICKNESS = 15
API_CALL_EVERY_N_FRAMES = 5        # Call API every 5 frames (not 15)
API_TIMEOUT = 2.0                   # 2 seconds — generous, but runs in background so it doesn't matter


# ═══════════════════════════════════════════════
# THREAD-SAFE GESTURE STATE
# ═══════════════════════════════════════════════

class GestureState:
    """
    Shared object between main thread (reads) and API thread (writes).
    Lock ensures no race conditions.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.gesture = None
        self.confidence = 0.0
        self.is_pending = False         # True jab API call chal rahi ho

    def update(self, gesture, confidence):
        with self._lock:
            self.gesture = gesture
            self.confidence = confidence
            self.is_pending = False

    def get(self):
        with self._lock:
            return self.gesture, self.confidence

    def mark_pending(self):
        with self._lock:
            self.is_pending = True

    def is_busy(self):
        with self._lock:
            return self.is_pending


gesture_state = GestureState()


# ═══════════════════════════════════════════════
# BACKGROUND API CALLER
# ═══════════════════════════════════════════════

def call_api_in_background(frame, lmList):
    """
    Ye function ek alag thread mein run hota hai.
    Main loop KABHI bhi yahan wait nahi karta.
    """
    try:
        h, w, _ = frame.shape

        xs = [pt[1] for pt in lmList]
        ys = [pt[2] for pt in lmList]

        x_min = max(0, min(xs) - 40)
        y_min = max(0, min(ys) - 40)
        x_max = min(w, max(xs) + 40)
        y_max = min(h, max(ys) + 40)

        crop = frame[y_min:y_max, x_min:x_max]

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            gesture_state.update(None, 0.0)
            return

        crop = cv2.resize(crop, (64, 64))
        _, buffer = cv2.imencode(".jpg", crop)

        response = requests.post(
            API_URL,
            files={"file": ("hand.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            gesture_state.update(data["gesture"], data["confidence"])
        else:
            gesture_state.update(None, 0.0)

    except requests.exceptions.Timeout:
        print("[API] Timeout — server slow hai, result skip.")
        gesture_state.update(None, 0.0)
    except requests.exceptions.ConnectionError:
        print("[API] Connection error — server chal raha hai kya?")
        gesture_state.update(None, 0.0)
    except Exception as e:
        print(f"[API] Unexpected error: {e}")
        gesture_state.update(None, 0.0)


def request_gesture_async(frame, lmList):
    """
    Ek naya daemon thread spawn karta hai API call ke liye.
    Main loop turant return kar leta hai — koi wait nahi.
    """
    if gesture_state.is_busy():
        return  # Pichli call abhi chal rahi hai, skip karo

    # Frame ki copy bhejo taaki main loop aage badh sake
    frame_copy = frame.copy()
    gesture_state.mark_pending()

    t = threading.Thread(
        target=call_api_in_background,
        args=(frame_copy, lmList),
        daemon=True     # Program close hone par thread apne aap band ho jaayega
    )
    t.start()


# ═══════════════════════════════════════════════
# HEADER / OVERLAY SETUP
# ═══════════════════════════════════════════════

folderPath = "header"
myList = sorted(os.listdir(folderPath))   # sorted for consistent ordering
overLayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    if image is not None:
        overLayList.append(image)

if not overLayList:
    raise RuntimeError(f"'{folderPath}' mein koi valid image nahi mili!")

header = overLayList[0]
drawcolor = (119, 0, 200)
brushthickness = BRUSH_THICKNESS
xp, yp = 0, 0

saved = False
thickness_locked = True

imgCanvas = np.zeros((720, 1280, 3), np.uint8)


# ═══════════════════════════════════════════════
# CAMERA + DETECTOR SETUP
# ═══════════════════════════════════════════════

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(min_detection_confidence=0.85)

frame_count = 0

print("Virtual Paint ready! 'q' dabao band karne ke liye.")


# ═══════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))
    img = cv2.flip(img, 1)

    # ── Hand Detection ──────────────────────────
    img = detector.findhands(img)
    lmList = detector.findposition(img, draw=False)

    if lmList:
        x1, y1 = lmList[8][1:]    # Index fingertip
        x2, y2 = lmList[12][1:]   # Middle fingertip
        x3, y3 = lmList[4][1:]    # Thumb tip
        x4, y4 = lmList[20][1:]   # Pinky tip

        fingers = detector.fingerUp()
        frame_count += 1

        # ── Async API Call (non-blocking) ────────
        if frame_count % API_CALL_EVERY_N_FRAMES == 0:
            request_gesture_async(img, lmList)

        # ── Get latest gesture from shared state ─
        gesture, confidence = gesture_state.get()

        # ── Fallback: use finger logic ONLY if model is uncertain ──
        # This fallback now only triggers when confidence is genuinely low,
        # not because the API is blocking/timing out.
        # Naya fallback — jab model confident nahi, MediaPipe sambhal lega
        if confidence < CONFIDENCE_THRESHOLD:
            gesture = None
            if fingers == [0, 1, 0, 0, 0]:
                gesture = "index"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "l"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "palm"
            elif fingers == [1, 1, 1, 1, 0]:
                gesture = "palm"
            elif fingers == [0, 0, 0, 0, 0]:
                gesture = "fist"
            elif fingers == [1, 1, 0, 0, 1]:
                gesture = "ok"  # approximate
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "thumb"

        # ── Gesture Actions ──────────────────────

        # CLEAR CANVAS
        if gesture == "palm":
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            xp, yp = 0, 0

        # SAVE IMAGE
        elif gesture == "ok":
            if not saved:
                cv2.imwrite("Drawing.png", imgCanvas)
                print("[Save] Drawing.png saved!")
                saved = True
            xp, yp = 0, 0

        # SELECTION MODE (2 fingers)
        elif gesture == "l":
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overLayList[0]
                    drawcolor = (119, 0, 200)
                    brushthickness = BRUSH_THICKNESS
                elif 550 < x1 < 750:
                    header = overLayList[1]
                    drawcolor = (0, 0, 255)
                elif 800 < x1 < 950:
                    header = overLayList[2]
                    drawcolor = (255, 0, 255)
                elif 1050 < x1 < 1200:
                    header = overLayList[3]
                    drawcolor = (0, 0, 0)
                    brushthickness = 50

        # THICKNESS ADJUST (Thumb + Index pinch)
        elif fingers == [1, 1, 0, 0, 0]:
            distance = math.hypot(x1 - x3, y1 - y3)
            if distance < 30:
                thickness_locked = False
            if not thickness_locked:
                brushthickness = int(np.interp(distance, [30, 180], [5, 40]))
                brushthickness = min(40, brushthickness)
            xp, yp = 0, 0

        # DRAWING MODE
        elif gesture == "index":
            if not thickness_locked:
                thickness_locked = True

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
            xp, yp = x1, y1

        else:
            xp, yp = 0, 0

        if gesture != "ok":
            saved = False

    # ── Compose Final Frame ──────────────────────

    img_gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 1, imgCanvas, 0.2, 0)

    # ── HUD / Debug Info ─────────────────────────

    gesture_display, conf_display = gesture_state.get()
    pending_indicator = " [...]" if gesture_state.is_busy() else ""

    cv2.putText(img, f'Thickness: {brushthickness}',
                (900, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # cv2.putText(img,
    #             f"{gesture_display or 'None'} {conf_display:.2f}{pending_indicator}",
    #             (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Virtual Paint', img)
    cv2.imshow('Canvas', imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Band ho gaya.")