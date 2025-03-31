import os
import sys
import time

from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# import tensorflow as tf # for TensorFlow

app = Flask(__name__)

# if true, run latency test
ENABLE_LATENCY_TEST = len(sys.argv) > 1 and sys.argv[1].lower() == "true"
print(f"[INFO] Latency test enabled: {ENABLE_LATENCY_TEST}")
inference_times = []
NUM_LATENCY_SAMPLES = 100

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models/20FrozenLayersRotationFlip.tflite"
)


# initialise MediaPipe
def init_mediapipe():
    global mp_hands, hands, mp_draw, mp_drawing, mp_drawing_styles

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


# load TFLite model
def load_tflite_model():
    global interpreter, input_details, output_details

    # interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) # for TensorFlow
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Loaded TFLite model: {MODEL_PATH}")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")


# initialise camera
def init_camera():
    global camera

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

# preprocess inference image to maintain aspect ratio
def maintain_aspect_ratio_resize(image, target_size):
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # calculate scaling factor
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # resize image
    resized = cv2.resize(image, (new_w, new_h))

    # create square black image
    square = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # calculate padding
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # place resized image in center
    square[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return square

# run inference on the image
def run_inference(image):
    global interpreter, input_details, output_details, inference_times

    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    if ENABLE_LATENCY_TEST:
        start_time = time.time()

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    if ENABLE_LATENCY_TEST:
        latency_ms = (time.time() - start_time) * 1000
        inference_times.append(latency_ms)

        if len(inference_times) == NUM_LATENCY_SAMPLES:
            avg_latency = sum(inference_times) / NUM_LATENCY_SAMPLES
            with open("tflite_inference_latency.txt", "w") as f:
                f.write(
                    f"Average latency over {NUM_LATENCY_SAMPLES} frames: {avg_latency:.2f} ms\n"
                )
            print(
                f"[Latency Test] Average latency: {avg_latency:.2f} ms (saved to tflite_inference_latency.txt)"
            )
            inference_times.clear()

    return output_data

# generate frames for video feed
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            # create a copy of the frame for the processed view
            display_frame = frame.copy()

            # convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process the frame and detect hands
            results = hands.process(rgb_frame)

            # create a black canvas for the processed ROI
            processed_view = np.zeros((224, 224, 3), dtype=np.uint8)

            # draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # get bounding box coordinates
                    h, w = frame.shape[:2]
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))

                    # add padding to bounding box
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)

                    # draw bounding box
                    cv2.rectangle(
                        display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA
                    )

                    # extract and process hand ROI
                    hand_roi = frame[y1:y2, x1:x2]
                    if hand_roi.size != 0:  # if ROI is not empty
                        # maintain preprocessing
                        processed_roi = maintain_aspect_ratio_resize(
                            hand_roi, (224, 224)
                        )
                        # store the processed ROI for display
                        processed_view = processed_roi.copy()

                        prediction = run_inference(processed_roi)

                        # get top 3 predictions
                        top3_idx = np.argsort(prediction[0])[-3:][::-1]
                        top3_letters = [chr(ord("a") + idx).upper() for idx in top3_idx]
                        top3_conf = [prediction[0][idx] for idx in top3_idx]

                        # draw top 3 predictions box
                        box_width = 200
                        box_height = 120
                        margin = 10

                        box_x = display_frame.shape[1] - box_width - margin
                        box_y = margin

                        overlay = display_frame.copy()
                        cv2.rectangle(
                            overlay,
                            (box_x, box_y),
                            (box_x + box_width, box_y + box_height),
                            (255, 255, 255),
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.addWeighted(
                            overlay, 0.7, display_frame, 0.3, 0, display_frame
                        )

                        cv2.putText(
                            display_frame,
                            "Top Predictions:",
                            (box_x + 5, box_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        # draw predictions
                        for i, (letter, conf) in enumerate(
                            zip(top3_letters, top3_conf)
                        ):
                            # green for top prediction, gray for others
                            bg_color = (0, 255, 0) if i == 0 else (200, 200, 200)

                            pred_y = box_y + 45 + i * 25
                            cv2.rectangle(
                                display_frame,
                                (box_x + 5, pred_y - 20),
                                (box_x + box_width - 5, pred_y + 5),
                                bg_color,
                                -1,
                                cv2.LINE_AA,
                            )

                            pred_text = f"{letter}: {conf:.2f}"
                            cv2.putText(
                                display_frame,
                                pred_text,
                                (box_x + 10, pred_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 0),
                                2,
                                cv2.LINE_AA,
                            )

                        # draw main prediction on the hand box
                        label = f"{top3_letters[0]}: {top3_conf[0]:.2f}"
                        label_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                        )[0]
                        cv2.rectangle(
                            display_frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (0, 255, 0),
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            display_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

            # resize display_frame to match height of processed view
            display_h = 224
            display_w = int(
                display_frame.shape[1] * (display_h / display_frame.shape[0])
            )
            display_frame = cv2.resize(display_frame, (display_w, display_h))

            combined_frame = np.hstack([display_frame, processed_view])

            # convert frame to jpg
            ret, buffer = cv2.imencode(".jpg", combined_frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


load_tflite_model()
init_camera()
init_mediapipe()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
