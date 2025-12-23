# python -m venv venv
# venv\Scripts\activate  (Windows)
# pip install -r requirements.txt

import os
import warnings
import logging
import platform
import copy
import itertools
import string

import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow import keras

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit

# -----------------------------
#  SUPPRESS WARNINGS
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

logging.getLogger('absl').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# TensorFlow availability flag (needed because keras comes from TF)
try:
    import tensorflow as tf  # noqa: F401
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not installed. Keras model (.h5) will not be available.")

# -----------------------------
#   FLASK + SOCKET
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
#   MODEL PATHS
# -----------------------------
SKELETON_MODEL_PATH = './model.p'
KERAS_MODEL_PATH = './model.h5'  # Indian sign / landmark-based model

# -----------------------------
#   LOAD SKELETON MODEL (model.p)
# -----------------------------
skeleton_model = None
try:
    model_dict = pickle.load(open(SKELETON_MODEL_PATH, 'rb'))
    skeleton_model = model_dict['model']
    print("✔ Skeleton model (model.p) loaded successfully")
except FileNotFoundError:
    print(f"⚠ Skeleton model not found at {SKELETON_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading skeleton model: {e}")

# -----------------------------
#   LOAD KERAS LANDMARK MODEL (.h5)
# -----------------------------
keras_model = None
if TENSORFLOW_AVAILABLE:
    try:
        if os.path.exists(KERAS_MODEL_PATH):
            # Try loading with compile=False first to avoid compilation issues
            try:
                keras_model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
                print("✔ Keras landmark model (model.h5) loaded successfully")
            except Exception as load_error:
                # Handle version compatibility issues with DepthwiseConv2D groups parameter
                if 'groups' in str(load_error) or 'DepthwiseConv2D' in str(load_error):
                    try:
                        # Create a custom DepthwiseConv2D that ignores the groups parameter
                        from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
                        
                        class CompatibleDepthwiseConv2D(BaseDepthwiseConv2D):
                            def __init__(self, *args, **kwargs):
                                # Remove 'groups' parameter if present (not supported in older TF versions)
                                kwargs.pop('groups', None)
                                super().__init__(*args, **kwargs)
                        
                        # Try loading with the custom object
                        keras_model = keras.models.load_model(
                            KERAS_MODEL_PATH, 
                            compile=False,
                            custom_objects={'DepthwiseConv2D': CompatibleDepthwiseConv2D}
                        )
                        print("✔ Keras landmark model (model.h5) loaded successfully (with compatibility fix)")
                    except Exception as e2:
                        print(f"⚠ Keras model loading failed: {e2}")
                        print("   The model may have been saved with a different TensorFlow version.")
                        print("   Continuing without Keras model - skeleton model will be used.")
                        keras_model = None
                else:
                    print(f"⚠ Keras model loading failed: {load_error}")
                    print("   Continuing without Keras model - skeleton model will be used.")
                    keras_model = None
        else:
            print(f"⚠ Keras model not found at {KERAS_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading Keras model: {e}")

# -----------------------------
#   LABELS / ALPHABET
# -----------------------------
# Labels for classical skeleton model (scikit-learn) - ASL + phrases
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Alphabet for Keras landmark model (digits + A–Z)
keras_alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Return model status for frontend badge."""
    return jsonify({
        'skeleton_model': skeleton_model is not None,
        'keras_model': keras_model is not None,
        'tensorflow': TENSORFLOW_AVAILABLE
    })


@socketio.on('connect')
def handle_connect():
    print("Client connected")


# ----------------------------------------------------
#   FIXED CAMERA OPENING (WINDOWS → CAP_DSHOW)
# ----------------------------------------------------
def open_camera():
    is_windows = platform.system().lower() == "windows"
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY

    print("Trying to open camera at index 0 using backend:", backend)

    cap = cv2.VideoCapture(0, backend)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("✔ Camera opened successfully at index 0")
            return cap
        else:
            print("⚠ Camera opened but no frames received.")
            cap.release()

    print("❌ ERROR: Could not access camera at index 0")
    return None


# ----------------------------------------------------
#   LANDMARK HELPERS (FROM REFERENCE SCRIPT)
# ----------------------------------------------------
def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to pixel coordinates list."""
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for lm in landmarks.landmark:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """Make landmarks relative to first point and normalize (reference logic)."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) or 1.0

    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


# ----------------------------------------------------
#   PREDICTION FUNCTIONS
# ----------------------------------------------------
def predict_with_skeleton_model(data_aux):
    """Make prediction using skeleton (MediaPipe landmarks) model"""
    if skeleton_model is None:
        return None, 0.0
    
    try:
        prediction = skeleton_model.predict([np.asarray(data_aux)])
        prediction_proba = skeleton_model.predict_proba([np.asarray(data_aux)])
        confidence = max(prediction_proba[0])
        predicted_label = labels_dict.get(int(prediction[0]), '?')
        return predicted_label, float(confidence)
    except Exception as e:
        print(f"Skeleton prediction error: {e}")
        return None, 0.0


def predict_with_cnn_model(frame, hand_region=None):
    """(Deprecated stub) kept for backward compatibility."""
    return None, 0.0


def predict_with_keras_landmark_model(pre_processed_landmark_list):
    """Make prediction using Keras landmark model (reference logic)."""
    if keras_model is None:
        return None, 0.0

    try:
        df = pd.DataFrame(pre_processed_landmark_list).transpose()
        predictions = keras_model.predict(df, verbose=0)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        if 0 <= predicted_class < len(keras_alphabet):
            label = keras_alphabet[predicted_class]
        else:
            label = '?'

        return label, confidence
    except Exception as e:
        print(f"Keras landmark model prediction error: {e}")
        return None, 0.0


def get_combined_prediction(skeleton_pred, skeleton_conf, keras_pred, keras_conf):
    """
    Combine predictions from both models.
    Uses the prediction with higher confidence, or ensemble if both are similar.
    """
    # If only one model has prediction, use that
    if skeleton_pred is None and keras_pred is not None:
        return keras_pred, keras_conf, "Keras"
    if keras_pred is None and skeleton_pred is not None:
        return skeleton_pred, skeleton_conf, "Skeleton"
    if skeleton_pred is None and keras_pred is None:
        return None, 0.0, None
    
    # Both models have predictions - use the one with higher confidence
    # Give slight preference to skeleton model as it's more reliable for hand gestures
    skeleton_weight = 1.1  # 10% boost for skeleton model
    
    weighted_skeleton_conf = skeleton_conf * skeleton_weight
    
    if weighted_skeleton_conf >= keras_conf:
        return skeleton_pred, skeleton_conf, "Skeleton"
    else:
        return keras_pred, keras_conf, "Keras"


# ----------------------------------------------------
#   MAIN FRAME GENERATOR
# ----------------------------------------------------
def generate_frames():
    cap = open_camera()

    # If camera not found → show placeholder forever
    if cap is None:
        print("Showing error placeholder frame...")

        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Not Available", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', error_frame)
        error_bytes = buffer.tobytes()

        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   error_bytes + b'\r\n')

    # --------------------------
    #  MEDIAPIPE INIT
    # --------------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0

    # ----------------------------------------------------
    #   FRAME LOOP
    # ----------------------------------------------------
    while True:
        try:
            data_aux, x_, y_ = [], [], []

            ret, frame = cap.read()

            if not ret or frame is None:
                print("⚠ Warning: Could not read frame.")
                continue

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            predicted_character = None
            confidence = 0.0
            model_used = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # --------------------------
                    #  SKELETON MODEL FEATURES
                    # --------------------------
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                    # --------------------------
                    #  BBOX FOR VISUALIZATION
                    # --------------------------
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    # --------------------------
                    #  KERAS LANDMARK FEATURES
                    # --------------------------
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    pre_processed_landmarks = pre_process_landmark(landmark_list)

                    # -------------------------------------
                    #  GET PREDICTIONS FROM BOTH MODELS
                    # -------------------------------------
                    skeleton_pred, skeleton_conf = predict_with_skeleton_model(data_aux)
                    keras_pred, keras_conf = predict_with_keras_landmark_model(pre_processed_landmarks)

                    # Combine predictions (ensemble)
                    predicted_character, confidence, model_used = get_combined_prediction(
                        skeleton_pred, skeleton_conf, keras_pred, keras_conf
                    )

                    # Send prediction via WebSocket
                    if predicted_character is not None:
                        socketio.emit(
                            'prediction',
                            {
                                'text': predicted_character, 
                                'confidence': float(confidence),
                                'model': model_used
                            }
                        )

                        # Draw bounding box and prediction
                        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(
                            frame,
                            f"{predicted_character} ({confidence*100:.1f}%)",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.1,
                            color,
                            3
                        )

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

        except Exception as e:
            print("Error in generate_frames:", e)
            break

    cap.release()
    hands.close()


@app.route('/video_feed')
def video_feed():
    print("Video feed accessed from", request.remote_addr)
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("\n" + "="*50)
    print("   SignSpeak - Sign Language Detection")
    print("="*50)
    print(f"Skeleton Model: {'✔ Loaded' if skeleton_model else '❌ Not Available'}")
    print(f"Keras Landmark Model: {'✔ Loaded' if keras_model else '❌ Not Available'}")
    print(f"TensorFlow/Keras: {'✔ Available' if TENSORFLOW_AVAILABLE else '❌ Not Installed'}")
    print("="*50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)