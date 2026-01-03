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
import time

import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow import keras
import hashlib
import csv
from functools import wraps

from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for, flash
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
#   AUTHENTICATION SETUP
# -----------------------------
DATABASE_CSV = 'database.csv'

def init_database():
    """Initialize the CSV database if it doesn't exist."""
    if not os.path.exists(DATABASE_CSV):
        with open(DATABASE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['username', 'password_hash', 'email'])

def hash_password(password):
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=''):
    """Register a new user in the CSV database."""
    init_database()
    
    # Check if username already exists
    if os.path.exists(DATABASE_CSV):
        df = pd.read_csv(DATABASE_CSV)
        if username in df['username'].values:
            return False, "Username already exists"
    
    # Add new user
    password_hash = hash_password(password)
    new_user = pd.DataFrame({
        'username': [username],
        'password_hash': [password_hash],
        'email': [email]
    })
    
    if os.path.exists(DATABASE_CSV):
        df = pd.read_csv(DATABASE_CSV)
        df = pd.concat([df, new_user], ignore_index=True)
    else:
        df = new_user
    
    df.to_csv(DATABASE_CSV, index=False)
    return True, "User registered successfully"

def verify_user(username, password):
    """Verify user credentials."""
    if not os.path.exists(DATABASE_CSV):
        return False, "Database not found"
    
    df = pd.read_csv(DATABASE_CSV)
    user = df[df['username'] == username]
    
    if user.empty:
        return False, "Username not found"
    
    password_hash = hash_password(password)
    stored_hash = user['password_hash'].values[0]
    
    if password_hash == stored_hash:
        return True, "Login successful"
    else:
        return False, "Incorrect password"

def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
#   MODEL PATHS
# -----------------------------
SKELETON_MODEL_PATH = './model/model.p'
KERAS_MODEL_PATH = './model/indian_sign_model.h5'  # Indian sign / landmark-based model

# -----------------------------
#   PREDICTION STABILITY CONFIGURATION
# -----------------------------
# Stability check: waits for sign to stabilize before predicting
# This improves accuracy by avoiding predictions while hand is still moving
PREDICTION_STABILITY_DURATION = 0.8  # seconds - how long same prediction must appear before emitting
PREDICTION_MIN_CONFIDENCE = 0.6  # minimum confidence threshold (0.0 to 1.0)
PREDICTION_BUFFER_SIZE = 30  # number of recent predictions to track for stability

# -----------------------------
#   LOAD SKELETON MODEL (model.p) - For skeleton detection only, not used for predictions
# -----------------------------
skeleton_model = None
try:
    model_dict = pickle.load(open(SKELETON_MODEL_PATH, 'rb'))
    skeleton_model = model_dict['model']
    print("✔ Skeleton model (model.p) loaded (for skeleton detection only)")
except FileNotFoundError:
    print(f"⚠ Skeleton model not found at {SKELETON_MODEL_PATH} (optional - not used for predictions)")
except Exception as e:
    print(f"⚠ Error loading skeleton model: {e} (optional - not used for predictions)")

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
                        print("   Predictions will not be available without the Keras model.")
                        keras_model = None
                else:
                    print(f"⚠ Keras model loading failed: {load_error}")
                    print("   Predictions will not be available without the Keras model.")
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
@login_required
def index():
    return render_template('index.html', username=session.get('username'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Sign up page."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        email = request.form.get('email', '').strip()
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('signup.html')
        
        success, message = register_user(username, password, email)
        if success:
            flash(message, 'success')
            return redirect(url_for('signin'))
        else:
            flash(message, 'error')
            return render_template('signup.html')
    
    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    """Sign in page."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('signin.html')
        
        success, message = verify_user(username, password)
        if success:
            session['username'] = username
            flash(message, 'success')
            return redirect(url_for('index'))
        else:
            flash(message, 'error')
            return render_template('signin.html')
    
    # If already logged in, redirect to home
    if 'username' in session:
        return redirect(url_for('index'))
    
    return render_template('signin.html')


@app.route('/logout')
def logout():
    """Logout and clear session."""
    username = session.pop('username', None)
    if username:
        flash(f'Logged out successfully, {username}', 'success')
    return redirect(url_for('signin'))


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
    
    # Try different backends based on platform
    if is_windows:
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        # For Linux/Docker, try V4L2 first, then fallback to ANY
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends_to_try:
        try:
            print(f"Trying to open camera at index 0 using backend: {backend}")
            cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✔ Camera opened successfully at index 0 with backend {backend}")
                    return cap
                else:
                    print(f"⚠ Camera opened with backend {backend} but no frames received.")
                    cap.release()
            else:
                print(f"⚠ Could not open camera with backend {backend}")
        except Exception as e:
            print(f"⚠ Error trying backend {backend}: {e}")
            continue
    
    print("❌ ERROR: Could not access camera at index 0 with any backend")
    print("   Note: In Docker, camera access requires proper device mounting.")
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
    """(Deprecated) Skeleton model is no longer used for predictions.
    Only indian_sign_model.h5 (Keras model) is used for predictions.
    Skeleton detection is handled by MediaPipe for visualization only."""
    return None, 0.0


def predict_with_cnn_model(frame, hand_region=None):
    """(Deprecated stub) kept for backward compatibility."""
    return None, 0.0


def predict_with_keras_landmark_model(pre_processed_landmark_list):
    """Make prediction using Keras landmark model (reference logic)."""
    if keras_model is None:
        return None, 0.0

    try:
        # Convert to numpy array and ensure correct shape
        # Model expects (batch_size, 84) - pad to 84 features if needed
        features = np.array(pre_processed_landmark_list, dtype=np.float32)
        
        # Pad to 84 features if we have less (for single hand)
        if len(features) < 84:
            # Pad with zeros to reach 84 features (model expects 84 for two-hand support)
            features = np.pad(features, (0, 84 - len(features)), mode='constant', constant_values=0.0)
        elif len(features) > 84:
            # Truncate if somehow we have more
            features = features[:84]
        
        # Reshape to (1, 84) for batch prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        predictions = keras_model.predict(features, verbose=0)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        if 0 <= predicted_class < len(keras_alphabet):
            label = keras_alphabet[predicted_class]
        else:
            label = '?'

        return label, confidence
    except Exception as e:
        print(f"Keras landmark model prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def get_combined_prediction(skeleton_pred, skeleton_conf, keras_pred, keras_conf):
    """
    (Deprecated) No longer combines predictions - only uses Keras model.
    This function is kept for backward compatibility but is not called.
    """
    # Only use Keras model for predictions
    if keras_pred is not None:
        return keras_pred, keras_conf, "Keras"
    return None, 0.0, None


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
    #   PREDICTION STABILITY CONFIGURATION
    # ----------------------------------------------------
    # Use global configuration values
    STABILITY_DURATION = PREDICTION_STABILITY_DURATION
    MIN_CONFIDENCE_THRESHOLD = PREDICTION_MIN_CONFIDENCE
    BUFFER_SIZE = PREDICTION_BUFFER_SIZE
    
    # Prediction stability tracking
    prediction_buffer = []  # Store recent predictions: [(character, confidence, timestamp), ...]
    last_emitted_prediction = None
    last_emitted_time = 0
    stable_prediction_start_time = None
    stable_prediction_character = None
    stable_prediction_confidence = 0.0

    # ----------------------------------------------------
    #   FRAME LOOP
    # ----------------------------------------------------
    consecutive_failures = 0
    max_failures = 10
    reopen_attempts = 0
    max_reopen_attempts = 3  # Limit how many times we try to reopen
    
    while True:
        try:
            data_aux, x_, y_ = [], [], []

            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures <= max_failures:
                    # Only print warning occasionally to avoid spam
                    if consecutive_failures % 5 == 0:
                        print(f"⚠ Warning: Could not read frame ({consecutive_failures} consecutive failures)")
                else:
                    # After many failures, try to reopen camera (but limit attempts)
                    if reopen_attempts < max_reopen_attempts:
                        reopen_attempts += 1
                        print(f"⚠ Multiple frame read failures. Attempting to reopen camera (attempt {reopen_attempts}/{max_reopen_attempts})...")
                        cap.release()
                        time.sleep(1.0)  # Longer pause before reopening
                        new_cap = open_camera()
                        if new_cap is not None:
                            cap = new_cap
                            consecutive_failures = 0
                            print("✔ Camera reopened successfully")
                            continue
                        else:
                            print(f"⚠ Could not reopen camera (attempt {reopen_attempts}/{max_reopen_attempts})")
                            # Try to recreate the original cap
                            cap = open_camera()
                            if cap is None:
                                print("❌ Could not reopen camera. Will continue trying to read frames...")
                                time.sleep(2.0)  # Wait longer before next attempt
                            else:
                                consecutive_failures = 0
                    else:
                        # After max reopen attempts, just log and continue
                        if consecutive_failures % 20 == 0:  # Print every 20 failures
                            print(f"⚠ Camera read failures persist ({consecutive_failures} failures, {reopen_attempts} reopen attempts). Continuing...")
                        time.sleep(0.1)  # Longer delay when camera is persistently unavailable
                
                # Small delay to avoid busy-waiting
                time.sleep(0.01)
                continue

            # Reset failure counter on successful read
            consecutive_failures = 0
            reopen_attempts = 0  # Reset reopen attempts on successful read

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            predicted_character = None
            confidence = 0.0
            model_used = None

            # Reset stability if no hands detected
            if not results.multi_hand_landmarks:
                stable_prediction_character = None
                stable_prediction_start_time = None
                stable_prediction_confidence = 0.0
            
            if results.multi_hand_landmarks:
                # Collect landmarks from all hands
                all_landmark_lists = []
                all_x_coords = []
                all_y_coords = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Collect coordinates for bbox
                    for lm in hand_landmarks.landmark:
                        all_x_coords.append(lm.x)
                        all_y_coords.append(lm.y)
                    
                    # Process landmarks for this hand
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    pre_processed = pre_process_landmark(landmark_list)
                    all_landmark_lists.append(pre_processed)

                # --------------------------
                #  BBOX FOR VISUALIZATION
                # --------------------------
                if all_x_coords and all_y_coords:
                    x1 = int(min(all_x_coords) * W) - 10
                    y1 = int(min(all_y_coords) * H) - 10
                    x2 = int(max(all_x_coords) * W) + 10
                    y2 = int(max(all_y_coords) * H) + 10

                    # --------------------------
                    #  COMBINE LANDMARKS FOR PREDICTION (model expects 84 features)
                    # --------------------------
                    # If single hand: pad with zeros to 84 features
                    # If two hands: concatenate both (42 + 42 = 84)
                    if len(all_landmark_lists) == 1:
                        # Single hand: pad with zeros
                        combined_landmarks = all_landmark_lists[0] + [0.0] * len(all_landmark_lists[0])
                    elif len(all_landmark_lists) >= 2:
                        # Two hands: concatenate first two hands
                        combined_landmarks = all_landmark_lists[0] + all_landmark_lists[1]
                    else:
                        combined_landmarks = None

                    # -------------------------------------
                    #  GET PREDICTION FROM KERAS MODEL ONLY
                    # -------------------------------------
                    if combined_landmarks is not None:
                        predicted_character, confidence = predict_with_keras_landmark_model(combined_landmarks)
                        model_used = "Keras"
                        current_time = time.time()

                        # Stability check: only emit if prediction is stable
                        if predicted_character is not None and confidence >= MIN_CONFIDENCE_THRESHOLD:
                            # Add to prediction buffer
                            prediction_buffer.append((predicted_character, confidence, current_time))
                            
                            # Keep buffer size manageable
                            if len(prediction_buffer) > BUFFER_SIZE:
                                prediction_buffer.pop(0)
                            
                            # Remove old predictions (older than stability duration)
                            prediction_buffer = [(char, conf, ts) for char, conf, ts in prediction_buffer 
                                               if current_time - ts <= STABILITY_DURATION + 0.5]
                            
                            # Check if current prediction matches stable prediction
                            if stable_prediction_character == predicted_character:
                                # Same prediction continues - check if stable long enough
                                if stable_prediction_start_time is not None:
                                    stability_duration = current_time - stable_prediction_start_time
                                    if stability_duration >= STABILITY_DURATION:
                                        # Prediction is stable - emit it (but only once per stable period)
                                        if (last_emitted_prediction != predicted_character or 
                                            current_time - last_emitted_time >= STABILITY_DURATION):
                                            socketio.emit(
                                                'prediction',
                                                {
                                                    'text': predicted_character, 
                                                    'confidence': float(stable_prediction_confidence),
                                                    'model': model_used
                                                }
                                            )
                                            last_emitted_prediction = predicted_character
                                            last_emitted_time = current_time
                            else:
                                # New prediction detected - reset stability tracking
                                stable_prediction_character = predicted_character
                                stable_prediction_confidence = confidence
                                stable_prediction_start_time = current_time
                            
                            # Draw bounding box and prediction (always show current prediction on video)
                            # Show stability status
                            stability_status = ""
                            if stable_prediction_start_time is not None:
                                elapsed = current_time - stable_prediction_start_time
                                if elapsed >= STABILITY_DURATION:
                                    stability_status = " [STABLE]"
                                else:
                                    stability_status = f" [{(elapsed/STABILITY_DURATION)*100:.0f}%]"
                            
                            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(
                                frame,
                                f"{predicted_character} ({confidence*100:.1f}%){stability_status}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.1,
                                color,
                                3
                            )
                        else:
                            # No valid prediction or low confidence - reset stability
                            stable_prediction_character = None
                            stable_prediction_start_time = None
                            stable_prediction_confidence = 0.0

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