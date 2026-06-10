# AI Virtual Painter 🎨

Paint in the air using hand gestures and computer vision!

## Demo
Control a virtual paintbrush using just your hand — no physical tools needed.

## Tech Stack
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/suhani1509/AI-VIRTUAL-PAINTER.git
cd AI-VIRTUAL-PAINTER
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the trained model
👉 [Download gesture_model.h5 from Google Drive](https://drive.google.com/drive/folders/1FYDTZJ86w6OIqAhViL2ucijTAPCvaOlh?usp=drive_link)
- `label_encoder.pkl` is included in the repo

Place `gesture_model.h5` in the project root folder.

### 4. Run the project
```bash
python virtualpaint.py
```

## How it works
1. Hand gestures are detected using MediaPipe
2. Gestures are classified using a CNN model trained on LeapGestRecog dataset
3. Based on the gesture — draw, erase, or change color!

## Model Training
Model was trained on Kaggle using GPU.
Training notebook and dataset: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)