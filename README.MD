# ASL Real-Time Inference App (Flask + TFLite + MediaPipe)

This application performs real-time American Sign Language (ASL) letter recognition using a TFLite model, OpenCV for webcam capture, and MediaPipe for hand tracking. The app runs in a Flask server and provides a live annotated video stream in the browser.

---

## Features

- Real-time hand detection using **MediaPipe**
- Model input preprocessing with aspect-ratio preservation
- Top-3 prediction overlay on video feed
- Optional latency testing for TFLite inference
- Simple HTML frontend for visualisation

---

## Requirements

Install dependencies using:

```pip install -r requirements.txt```

---

## Running the app

```python app.py```

By default, the app runs at:
http://127.0.0.1:5000

To enable latency measurement (average inference time/100 frames)

```python app.py true```

Latency is displayed in the terminal and saved to tflite_inference_latency.txt.

---

## Model
The model is located in:
models/20FrozenLayersRotationFlip.tflite

The hand region is:
•	Cropped using MediaPipe landmarks
•	Resized to 224×224 with padding to preserve aspect ratio
•	Normalised and passed into the TFLite model
