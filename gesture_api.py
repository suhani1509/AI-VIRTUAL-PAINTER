"""
gesture_api.py — Optimized FastAPI Gesture Recognition Server
=============================================================
Chalane ka tarika (alag terminal mein):
    uvicorn gesture_api:app --host 0.0.0.0 --port 8000

Fixes applied:
    1. Grayscale input (1 channel) — model ka actual input shape hai (None,64,64,1)
    2. TF_DISABLE_MKL=1 — grouped convolution CPU crash fix
    3. Keyword argument for infer() — positional arg TypeError fix
    4. Model warm-up on startup
"""

import os
# FIX: Grouped convolution CPU crash — TensorFlow import se PEHLE set karna zaroori hai
os.environ["TF_DISABLE_MKL"] = "1"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
import uvicorn

# ═══════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════

CLASSES = [
    'palm', 'l', 'fist', 'fist_moved', 'thumb',
    'index', 'ok', 'palm_moved', 'c', 'down'
]
IMG_SIZE = 64
MODEL_PATH = "gesture_saved_model"


# ═══════════════════════════════════════
# MODEL (loaded once, reused forever)
# ═══════════════════════════════════════

model = None
infer = None
input_key = None   # Model ka exact input key — runtime par discover hoti hai


def load_model_and_warmup():
    """
    Model load karo + ek dummy prediction chalao.
    Warm-up ensure karta hai ki pehli real request par
    TensorFlow ka graph compilation overhead na ho.
    """
    global model, infer, input_key

    print("[Server] Model load ho raha hai...")
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]

    # Model ka actual input key discover karo (hardcode mat karo)
    input_key = list(infer.structured_input_signature[1].keys())[0]
    print(f"[Server] Model input key: '{input_key}'")

    # Model ka actual input shape check karo
    input_spec = infer.structured_input_signature[1][input_key]
    print(f"[Server] Model input shape: {input_spec.shape}  dtype: {input_spec.dtype}")

    # Warm-up: shape se channels automatically detect karo
    # Error mein dikh raha tha: shape=(None, 64, 64, 1) — grayscale!
    channels = input_spec.shape[-1] if input_spec.shape.rank == 4 else 1
    dummy = tf.constant(
        np.zeros((1, IMG_SIZE, IMG_SIZE, channels), dtype=np.float32)
    )

    # FIX: keyword argument use karo (positional se TypeError aata tha)
    _ = infer(**{input_key: dummy})
    print("[Server] Model warm-up complete — ready to predict!")


# ═══════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_warmup()
    yield
    print("[Server] Shutting down.")


# ═══════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════

app = FastAPI(
    title="Gesture Recognition API",
    version="3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "message": "Gesture API v3 chal rahi hai!",
        "classes": CLASSES,
        "model_loaded": infer is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": infer is not None}


@app.post("/predict", response_model=GestureResponse)
async def predict(file: UploadFile = File(...)):
    if infer is None:
        return GestureResponse(gesture="unknown", confidence=0.0)

    # ── Decode image ─────────────────────────────
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return GestureResponse(gesture="unknown", confidence=0.0)

    # ── Preprocess ───────────────────────────────
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # FIX: Model grayscale (1 channel) expect karta hai
    # Training mein bhi grayscale use hua hoga
    gray       = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0

    # Shape: (1, 64, 64, 1)
    input_tensor = tf.constant(normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1))

    # ── Inference (keyword arg — positional se crash hota tha) ───
    output      = infer(**{input_key: input_tensor})
    predictions = list(output.values())[0].numpy()[0]

    class_idx  = int(np.argmax(predictions))
    confidence = float(predictions[class_idx])
    gesture    = CLASSES[class_idx]

    return GestureResponse(gesture=gesture, confidence=confidence)


# ═══════════════════════════════════════
# DIRECT RUN
# ═══════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "gesture_api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False
    )