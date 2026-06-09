"""
gesture_api.py — FastAPI Gesture Recognition Server
=====================================================
Chalane ka tarika (alag terminal mein):
    uvicorn gesture_api:app --host 0.0.0.0 --port 8000 --reload

Test karne ke liye browser mein:
    http://localhost:8000
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
import pickle
import uvicorn

# ═══════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════

app = FastAPI(title="Gesture Recognition API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════
# MODEL LOAD — server start hote hi ek baar
# ═══════════════════════════════════════

print("Model load ho raha hai...")

model = tf.keras.models.load_model("gesture_model.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print(f"Model ready! Classes: {list(le.classes_)}")

# ═══════════════════════════════════════
# RESPONSE SCHEMA
# ═══════════════════════════════════════

class GestureResponse(BaseModel):
    gesture: str
    confidence: float

# ═══════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Gesture API chal rahi hai!",
        "classes": list(le.classes_)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=GestureResponse)
async def predict(file: UploadFile = File(...)):
    """
    Hand ka cropped image bhejo -> gesture naam wapas milega.
    virtualpaint.py yahan POST request karega.
    """
    # Image bytes read karo
    contents  = await file.read()
    nparr     = np.frombuffer(contents, np.uint8)
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return GestureResponse(gesture="unknown", confidence=0.0)

    # Preprocess — training jaisa hi
    resized    = cv2.resize(img, (64, 64))
    normalized = resized.astype("float32") / 255.0
    input_arr  = np.expand_dims(normalized, axis=0)  # (1, 64, 64, 3)

    # Predict
    predictions = model.predict(input_arr, verbose=0)[0]
    class_idx   = int(np.argmax(predictions))
    confidence  = float(predictions[class_idx])
    gesture     = le.inverse_transform([class_idx])[0]

    return GestureResponse(gesture=gesture, confidence=confidence)


# ═══════════════════════════════════════
# DIRECT RUN
# ═══════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)