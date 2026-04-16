"""
API Routes Module — Cloud-Ready Architecture
Accepts base64 frames from client-side camera, processes with MediaPipe,
returns predictions. Works both locally and on cloud deployment.
"""

import cv2
import time
import json
import base64
import logging
import threading
import numpy as np
from flask import Blueprint, Response, jsonify, request

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW,
    PREDICTION_COOLDOWN, SIGN_CLASSES,
    MEDIAPIPE_MAX_HANDS, MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    WORD_DICTIONARY_PATH, CUSTOM_SIGNS_PATH,
    DEFAULT_DETECTION_MODE, WORD_SIGN_AUTO_SPACE,
    PREDICTION_SKIP_FRAMES, MAJORITY_VOTE_RATIO,
    STABILITY_REQUIRED_FRAMES,
)
from utils.hand_tracker import HandTracker
from utils.smoothing import PredictionSmoother
from utils.word_engine import WordEngine
from model.predict import GesturePredictor

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)

# ─── Global State ────────────────────────────────────────────

# Local-mode camera (optional, for local development)
camera = None
camera_lock = threading.Lock()
is_streaming = False

tracker = None
predictor = None
smoother = None
word_engine = None

# Sentence state
current_sentence = ""
last_prediction = ""
last_add_time = 0
current_prediction_label = ""
current_confidence = 0.0
current_category = ""
detected_words_history = []
current_partial_word = ""

# Detection mode
detection_mode = DEFAULT_DETECTION_MODE

# Auto-speak setting
auto_speak_enabled = False

# Frame counter for skip optimization
frame_counter = 0

# TTS availability (server-side pyttsx3 is optional)
tts_available = False
tts_engine = None
tts_lock = threading.Lock()


def init_components():
    """Initialize all detection components."""
    global tracker, predictor, smoother, word_engine, tts_available
    tracker = HandTracker(
        max_hands=MEDIAPIPE_MAX_HANDS,
        min_detection_conf=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_conf=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
    predictor = GesturePredictor()
    smoother = PredictionSmoother(
        window_size=SMOOTHING_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        majority_ratio=MAJORITY_VOTE_RATIO,
        stability_frames=STABILITY_REQUIRED_FRAMES,
    )
    word_engine = WordEngine(dictionary_path=WORD_DICTIONARY_PATH)

    # Check if pyttsx3 is available (local mode only)
    try:
        import pyttsx3
        tts_available = True
    except ImportError:
        tts_available = False
        logger.info("pyttsx3 not available — TTS handled client-side via Web Speech API")

    logger.info(
        f"All components initialized | "
        f"Predictor: ready | "
        f"WordEngine: {len(word_engine.words)} words | "
        f"Mode: {DEFAULT_DETECTION_MODE} | "
        f"TTS server-side: {tts_available}"
    )


# ═══════════════════════════════════════════════════════════════
#   CLOUD-READY: Client-side camera frame processing
# ═══════════════════════════════════════════════════════════════

@api.route("/process_frame", methods=["POST"])
def api_process_frame():
    """
    Process a single frame from the client's camera.
    
    Expects JSON: { "frame": "<base64-encoded JPEG>" }
    Returns JSON: { "prediction", "confidence", "category", "sentence", ... }
    """
    global current_prediction_label, current_confidence, current_category
    global current_sentence, last_prediction, last_add_time
    global current_partial_word, detected_words_history, frame_counter

    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400

    frame_data = request.json.get("frame", "")
    if not frame_data:
        return jsonify({"error": "No frame data"}), 400

    try:
        # Decode base64 → image
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]  # Strip data:image/jpeg;base64,

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        frame_counter += 1

        # Detect hands (no drawing needed — client handles display)
        _, results = tracker.find_hands(frame, draw=False)
        landmarks = tracker.extract_landmarks(results)

        prediction_result = {
            "prediction": current_prediction_label,
            "confidence": round(current_confidence, 4),
            "category": current_category,
            "sentence": current_sentence,
            "partial_word": current_partial_word,
            "model_ready": predictor.is_ready if predictor else False,
            "mode": detection_mode,
            "hand_detected": landmarks is not None,
            "landmarks": landmarks.tolist() if landmarks is not None else None,
            "suggestions": {},
        }

        if landmarks is not None and predictor.is_ready:
            label, conf = predictor.predict(landmarks)

            if label is not None:
                smoother.add_prediction(label, conf)
                smoothed_label, smoothed_conf = smoother.get_smoothed_prediction()

                if smoothed_label is not None:
                    current_prediction_label = smoothed_label
                    current_confidence = smoothed_conf
                    current_category = predictor.last_category or ""

                    # Add to sentence with cooldown
                    now = time.time()
                    if smoother.is_new_prediction(smoothed_label) or \
                       smoother.time_since_last_accept() > PREDICTION_COOLDOWN * 2:
                        if smoother.time_since_last_accept() > PREDICTION_COOLDOWN:
                            _process_detection(smoothed_label, now)
                            smoother.mark_accepted(smoothed_label)
        else:
            if landmarks is None:
                current_prediction_label = ""
                current_confidence = 0.0
                current_category = ""

        # Update response with latest state
        prediction_result.update({
            "prediction": current_prediction_label,
            "confidence": round(current_confidence, 4),
            "category": current_category,
            "sentence": current_sentence,
            "partial_word": current_partial_word,
        })

        # Add word suggestions
        if word_engine:
            prediction_result["suggestions"] = word_engine.get_suggestions(
                current_sentence, current_partial_word
            )

        return jsonify(prediction_result)

    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
#   LOCAL MODE: Server-side camera (for local development)
# ═══════════════════════════════════════════════════════════════

def start_camera():
    """Open the local webcam (dev mode only)."""
    global camera, is_streaming
    with camera_lock:
        if camera is not None and camera.isOpened():
            return True
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            logger.error("Failed to open camera")
            camera = None
            return False
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        is_streaming = True
        logger.info("Local camera started")
        return True


def stop_camera():
    """Release the local webcam."""
    global camera, is_streaming
    with camera_lock:
        is_streaming = False
        if camera is not None:
            camera.release()
            camera = None
        logger.info("Local camera stopped")


def generate_frames():
    """Generator yielding MJPEG frames (local mode only)."""
    global current_prediction_label, current_confidence, current_category
    global current_sentence, last_prediction, last_add_time
    global current_partial_word, detected_words_history, frame_counter

    while is_streaming:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            ret, frame = camera.read()

        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        frame, results = tracker.find_hands(frame, draw=True)

        if frame_counter % PREDICTION_SKIP_FRAMES == 0:
            landmarks = tracker.extract_landmarks(results)

            if landmarks is not None and predictor.is_ready:
                label, conf = predictor.predict(landmarks)

                if label is not None:
                    smoother.add_prediction(label, conf)
                    smoothed_label, smoothed_conf = smoother.get_smoothed_prediction()

                    if smoothed_label is not None:
                        current_prediction_label = smoothed_label
                        current_confidence = smoothed_conf
                        current_category = predictor.last_category or ""

                        now = time.time()
                        if smoother.is_new_prediction(smoothed_label) or \
                           smoother.time_since_last_accept() > PREDICTION_COOLDOWN * 2:
                            if smoother.time_since_last_accept() > PREDICTION_COOLDOWN:
                                _process_detection(smoothed_label, now)
                                smoother.mark_accepted(smoothed_label)
            else:
                current_prediction_label = ""
                current_confidence = 0.0
                current_category = ""

        if current_prediction_label:
            cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 60), (0, 0, 0), -1)
            cat_text = f" [{current_category}]" if current_category else ""
            cv2.putText(
                frame,
                f"{current_prediction_label}  {current_confidence:.0%}{cat_text}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2,
            )

        bboxes = tracker.get_hand_bbox(frame, results)
        for (x, y, w, h) in bboxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 200), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )
        time.sleep(0.01)


# ─── Process Detection ───────────────────────────────────────

def _process_detection(label, now):
    """Process a confirmed detection and add to sentence."""
    global current_sentence, last_prediction, last_add_time
    global current_partial_word, detected_words_history

    is_word_sign = len(label) > 1 and label not in ('SPACE', 'DELETE')

    if label == "SPACE":
        if current_partial_word and word_engine:
            final_word, corrected, original = word_engine.complete_word()
            if corrected:
                current_sentence = current_sentence[:-len(original)] + final_word
            current_partial_word = ""
        current_sentence += " "

    elif label == "DELETE":
        if current_partial_word:
            current_partial_word = current_partial_word[:-1]
            if word_engine:
                word_engine.current_word_buffer = current_partial_word
        current_sentence = current_sentence[:-1] if current_sentence else ""

    elif is_word_sign:
        if current_sentence and not current_sentence.endswith(" "):
            current_sentence += " "
        current_sentence += label
        if WORD_SIGN_AUTO_SPACE:
            current_sentence += " "
        current_partial_word = ""
        if word_engine:
            word_engine.reset_buffer()
        detected_words_history.append({
            'word': label, 'time': now, 'category': current_category,
        })
    else:
        current_sentence += label
        current_partial_word += label
        if word_engine:
            word_engine.add_letter(label)

    last_prediction = label
    last_add_time = now


def _speak_async(text):
    """Speak text in background thread (local mode only)."""
    if not tts_available:
        return  # Client handles TTS via Web Speech API
    def _speak():
        global tts_engine
        with tts_lock:
            try:
                import pyttsx3
                if tts_engine is None:
                    tts_engine = pyttsx3.init()
                    tts_engine.setProperty('rate', 150)
                    tts_engine.setProperty('volume', 0.9)
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
                tts_engine = None
    threading.Thread(target=_speak, daemon=True).start()


# ─── API Endpoints ───────────────────────────────────────────

@api.route("/video_feed")
def video_feed():
    """Stream webcam feed as MJPEG (local mode)."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@api.route("/start_camera", methods=["POST"])
def api_start_camera():
    """Start local camera (local mode)."""
    success = start_camera()
    if success:
        return jsonify({"status": "ok", "message": "Camera started", "mode": "local"})
    return jsonify({"status": "error", "message": "Failed to start camera"}), 500


@api.route("/stop_camera", methods=["POST"])
def api_stop_camera():
    stop_camera()
    return jsonify({"status": "ok", "message": "Camera stopped"})


@api.route("/predict", methods=["GET"])
def api_predict():
    """Get current prediction state."""
    suggestions = {}
    if word_engine:
        suggestions = word_engine.get_suggestions(current_sentence, current_partial_word)
    return jsonify({
        "prediction": current_prediction_label,
        "confidence": round(current_confidence, 4),
        "category": current_category,
        "sentence": current_sentence,
        "partial_word": current_partial_word,
        "model_ready": predictor.is_ready if predictor else False,
        "mode": detection_mode,
        "suggestions": suggestions,
        "tts_server": tts_available,
    })


@api.route("/reset", methods=["POST"])
def api_reset():
    """Clear everything."""
    global current_sentence, last_prediction, last_add_time
    global current_prediction_label, current_confidence
    global current_partial_word, detected_words_history, current_category
    current_sentence = ""
    last_prediction = ""
    last_add_time = 0
    current_prediction_label = ""
    current_confidence = 0.0
    current_category = ""
    current_partial_word = ""
    detected_words_history = []
    if smoother:
        smoother.reset()
    if word_engine:
        word_engine.reset_buffer()
    return jsonify({"status": "ok", "message": "Reset complete"})


@api.route("/speak", methods=["POST"])
def api_speak():
    """Server-side TTS (local mode). Returns tts_server flag for client fallback."""
    text = request.json.get("text", current_sentence) if request.is_json else current_sentence
    if not text.strip():
        return jsonify({"status": "error", "message": "Nothing to speak"}), 400
    if tts_available:
        _speak_async(text)
        return jsonify({"status": "ok", "message": f"Speaking: {text}", "tts_server": True})
    return jsonify({"status": "ok", "message": text, "tts_server": False})


@api.route("/status", methods=["GET"])
def api_status():
    return jsonify({
        "camera_active": is_streaming,
        "model_ready": predictor.is_ready if predictor else False,
        "classes": len(SIGN_CLASSES),
        "current_sentence": current_sentence,
        "mode": detection_mode,
        "auto_speak": auto_speak_enabled,
        "tts_server": tts_available,
    })


@api.route("/mode", methods=["POST"])
def api_set_mode():
    global detection_mode
    mode = request.json.get("mode", "auto") if request.is_json else "auto"
    if predictor:
        detection_mode = predictor.set_mode(mode)
    return jsonify({"status": "ok", "mode": detection_mode})


@api.route("/auto_speak", methods=["POST"])
def api_toggle_auto_speak():
    global auto_speak_enabled
    auto_speak_enabled = not auto_speak_enabled
    return jsonify({"status": "ok", "auto_speak": auto_speak_enabled})


@api.route("/suggestions", methods=["GET"])
def api_suggestions():
    if not word_engine:
        return jsonify({"suggestions": {}})
    return jsonify({
        "suggestions": word_engine.get_suggestions(current_sentence, current_partial_word),
    })


@api.route("/complete_word", methods=["POST"])
def api_complete_word():
    global current_sentence, current_partial_word
    word = request.json.get("word", "") if request.is_json else ""
    if word and current_partial_word:
        current_sentence = current_sentence[:-len(current_partial_word)] + word + " "
        current_partial_word = ""
        if word_engine:
            word_engine.reset_buffer()
    return jsonify({"status": "ok", "sentence": current_sentence})


@api.route("/insert_word", methods=["POST"])
def api_insert_word():
    global current_sentence, current_partial_word
    word = request.json.get("word", "") if request.is_json else ""
    if word:
        if current_sentence and not current_sentence.endswith(" "):
            current_sentence += " "
        current_sentence += word + " "
        current_partial_word = ""
        if word_engine:
            word_engine.reset_buffer()
    return jsonify({"status": "ok", "sentence": current_sentence})


@api.route("/library", methods=["GET"])
def api_library():
    if not predictor:
        return jsonify({"signs": [], "categories": []})
    return jsonify(predictor.get_library_info())


@api.route("/library/search", methods=["GET"])
def api_library_search():
    query = request.args.get("q", "")
    if not query or not predictor:
        return jsonify({"results": []})
    return jsonify({"results": predictor.search_library(query)})


@api.route("/library/categories", methods=["GET"])
def api_library_categories():
    if not predictor:
        return jsonify({"categories": []})
    return jsonify({"categories": predictor.get_library_info()['categories']})


@api.route("/add_sign", methods=["POST"])
def api_add_sign():
    data = request.json if request.is_json else {}
    label = data.get("label", "").strip().upper()
    description = data.get("description", "")
    category = data.get("category", "custom")
    if not label:
        return jsonify({"status": "error", "message": "Label is required"}), 400
    try:
        import os
        custom = {"signs": {}}
        if CUSTOM_SIGNS_PATH and os.path.exists(CUSTOM_SIGNS_PATH):
            with open(CUSTOM_SIGNS_PATH, 'r') as f:
                custom = json.load(f)
        custom['signs'][label] = {
            "description": description, "category": category,
            "type": "custom", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(CUSTOM_SIGNS_PATH, 'w') as f:
            json.dump(custom, f, indent=2)
        return jsonify({"status": "ok", "message": f"Sign '{label}' saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api.route("/add_char", methods=["POST"])
def api_add_char():
    global current_sentence, current_partial_word
    char = request.json.get("char", "") if request.is_json else ""
    if char:
        current_sentence += char
        if char != " ":
            current_partial_word += char
        else:
            current_partial_word = ""
    return jsonify({"status": "ok", "sentence": current_sentence})


@api.route("/backspace", methods=["POST"])
def api_backspace():
    global current_sentence, current_partial_word
    if current_partial_word:
        current_partial_word = current_partial_word[:-1]
    current_sentence = current_sentence[:-1] if current_sentence else ""
    return jsonify({"status": "ok", "sentence": current_sentence})


@api.route("/word_history", methods=["GET"])
def api_word_history():
    return jsonify({"history": detected_words_history[-20:], "total": len(detected_words_history)})


@api.route("/history", methods=["GET"])
def api_history():
    return jsonify({"history": [current_sentence] if current_sentence else []})
