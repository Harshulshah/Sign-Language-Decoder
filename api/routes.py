"""
API Routes Module
Defines Flask API endpoints for video streaming, prediction, word engine,
sign library, custom signs, TTS, and system management.
"""

import cv2
import time
import json
import logging
import threading
import pyttsx3
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
    PREDICTION_SKIP_FRAMES,
)
from utils.hand_tracker import HandTracker
from utils.smoothing import PredictionSmoother
from utils.word_engine import WordEngine
from model.predict import GesturePredictor

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)

# ─── Global State ────────────────────────────────────────────

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

# TTS engine (initialized lazily)
tts_engine = None
tts_lock = threading.Lock()

# Frame counter for skip optimization
frame_counter = 0


def init_components():
    """Initialize all components."""
    global tracker, predictor, smoother, word_engine
    tracker = HandTracker(
        max_hands=MEDIAPIPE_MAX_HANDS,
        min_detection_conf=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_conf=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
    predictor = GesturePredictor()
    smoother = PredictionSmoother(
        window_size=SMOOTHING_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    word_engine = WordEngine(dictionary_path=WORD_DICTIONARY_PATH)

    logger.info(
        f"All components initialized | "
        f"Predictor: ready | "
        f"WordEngine: {len(word_engine.words)} words | "
        f"Mode: {DEFAULT_DETECTION_MODE}"
    )


# ─── Camera Management ───────────────────────────────────────

def start_camera():
    """Open the webcam."""
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
        logger.info("Camera started")
        return True


def stop_camera():
    """Release the webcam."""
    global camera, is_streaming
    with camera_lock:
        is_streaming = False
        if camera is not None:
            camera.release()
            camera = None
        logger.info("Camera stopped")


def generate_frames():
    """
    Generator yielding MJPEG frames with hand overlay.
    Runs prediction on each frame using hybrid classifier.
    """
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

        # Detect hands and draw landmarks
        frame, results = tracker.find_hands(frame, draw=True)

        # Run prediction (with optional frame skipping)
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

                        # Add to sentence with cooldown
                        now = time.time()
                        if (smoothed_label != last_prediction or
                                now - last_add_time > PREDICTION_COOLDOWN * 2):
                            if now - last_add_time > PREDICTION_COOLDOWN:
                                _process_detection(smoothed_label, now)
                else:
                    pass  # Keep last prediction displayed briefly
            else:
                current_prediction_label = ""
                current_confidence = 0.0
                current_category = ""

        # Draw prediction overlay on frame
        if current_prediction_label:
            cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 60), (0, 0, 0), -1)
            cat_text = f" [{current_category}]" if current_category else ""
            cv2.putText(
                frame,
                f"{current_prediction_label}  {current_confidence:.0%}{cat_text}",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 200), 2,
            )

        # Draw hand bounding boxes
        bboxes = tracker.get_hand_bbox(frame, results)
        for (x, y, w, h) in bboxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 200), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        time.sleep(0.01)


def _process_detection(label, now):
    """Process a confirmed detection and add to sentence."""
    global current_sentence, last_prediction, last_add_time
    global current_partial_word, detected_words_history

    is_word_sign = len(label) > 1 and label not in ('SPACE', 'DELETE')

    if label == "SPACE":
        # Complete current partial word
        if current_partial_word and word_engine:
            final_word, corrected, original = word_engine.complete_word()
            if corrected:
                # Replace partial with corrected version
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
        # Detected a full word — add with auto-spacing
        if current_sentence and not current_sentence.endswith(" "):
            current_sentence += " "
        current_sentence += label
        if WORD_SIGN_AUTO_SPACE:
            current_sentence += " "
        current_partial_word = ""
        if word_engine:
            word_engine.reset_buffer()
        detected_words_history.append({
            'word': label,
            'time': now,
            'category': current_category,
        })
        # Auto-speak word if enabled
        if auto_speak_enabled:
            _speak_async(label)

    else:
        # Single letter/number — add to word buffer
        current_sentence += label
        current_partial_word += label
        if word_engine:
            word_engine.add_letter(label)

    last_prediction = label
    last_add_time = now


def _speak_async(text):
    """Speak text in background thread."""
    def _speak():
        global tts_engine
        with tts_lock:
            try:
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
    """Stream webcam feed as MJPEG."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@api.route("/start_camera", methods=["POST"])
def api_start_camera():
    success = start_camera()
    if success:
        return jsonify({"status": "ok", "message": "Camera started"})
    return jsonify({"status": "error", "message": "Failed to start camera"}), 500


@api.route("/stop_camera", methods=["POST"])
def api_stop_camera():
    stop_camera()
    return jsonify({"status": "ok", "message": "Camera stopped"})


@api.route("/predict", methods=["GET"])
def api_predict():
    """Get current prediction, sentence, and suggestions."""
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
    """Convert sentence to speech."""
    text = request.json.get("text", current_sentence) if request.is_json else current_sentence
    if not text.strip():
        return jsonify({"status": "error", "message": "Nothing to speak"}), 400
    _speak_async(text)
    return jsonify({"status": "ok", "message": f"Speaking: {text}"})


@api.route("/status", methods=["GET"])
def api_status():
    """Get system status."""
    return jsonify({
        "camera_active": is_streaming,
        "model_ready": predictor.is_ready if predictor else False,
        "classes": len(SIGN_CLASSES),
        "current_sentence": current_sentence,
        "mode": detection_mode,
        "auto_speak": auto_speak_enabled,
    })


# ─── New Endpoints ───────────────────────────────────────────

@api.route("/mode", methods=["POST"])
def api_set_mode():
    """Set detection mode: auto, alphabet, words."""
    global detection_mode
    mode = request.json.get("mode", "auto") if request.is_json else "auto"
    if predictor:
        detection_mode = predictor.set_mode(mode)
    return jsonify({"status": "ok", "mode": detection_mode})


@api.route("/auto_speak", methods=["POST"])
def api_toggle_auto_speak():
    """Toggle auto-speak for detected words."""
    global auto_speak_enabled
    auto_speak_enabled = not auto_speak_enabled
    return jsonify({"status": "ok", "auto_speak": auto_speak_enabled})


@api.route("/suggestions", methods=["GET"])
def api_suggestions():
    """Get word suggestions for current partial input."""
    if not word_engine:
        return jsonify({"suggestions": {}})
    return jsonify({
        "suggestions": word_engine.get_suggestions(current_sentence, current_partial_word),
    })


@api.route("/complete_word", methods=["POST"])
def api_complete_word():
    """Complete current partial word with a suggestion."""
    global current_sentence, current_partial_word
    word = request.json.get("word", "") if request.is_json else ""
    if word and current_partial_word:
        # Replace partial with completed word
        current_sentence = current_sentence[:-len(current_partial_word)] + word + " "
        current_partial_word = ""
        if word_engine:
            word_engine.reset_buffer()
    return jsonify({"status": "ok", "sentence": current_sentence})


@api.route("/insert_word", methods=["POST"])
def api_insert_word():
    """Insert a word (from suggestion or library) into the sentence."""
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
    """Get the full sign library."""
    if not predictor:
        return jsonify({"signs": [], "categories": []})
    info = predictor.get_library_info()
    return jsonify(info)


@api.route("/library/search", methods=["GET"])
def api_library_search():
    """Search the sign library."""
    query = request.args.get("q", "")
    if not query or not predictor:
        return jsonify({"results": []})
    results = predictor.search_library(query)
    return jsonify({"results": results})


@api.route("/library/categories", methods=["GET"])
def api_library_categories():
    """Get sign categories."""
    if not predictor:
        return jsonify({"categories": []})
    info = predictor.get_library_info()
    return jsonify({"categories": info['categories']})


@api.route("/add_sign", methods=["POST"])
def api_add_sign():
    """Save a custom sign definition."""
    data = request.json if request.is_json else {}
    label = data.get("label", "").strip().upper()
    description = data.get("description", "")
    category = data.get("category", "custom")

    if not label:
        return jsonify({"status": "error", "message": "Label is required"}), 400

    try:
        # Load existing custom signs
        custom = {"signs": {}}
        if CUSTOM_SIGNS_PATH:
            import os
            if os.path.exists(CUSTOM_SIGNS_PATH):
                with open(CUSTOM_SIGNS_PATH, 'r') as f:
                    custom = json.load(f)

        custom['signs'][label] = {
            "description": description,
            "category": category,
            "type": "custom",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(CUSTOM_SIGNS_PATH, 'w') as f:
            json.dump(custom, f, indent=2)

        logger.info(f"Custom sign saved: {label}")
        return jsonify({"status": "ok", "message": f"Sign '{label}' saved"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api.route("/word_history", methods=["GET"])
def api_word_history():
    """Get history of detected words."""
    return jsonify({
        "history": detected_words_history[-20:],  # Last 20
        "total": len(detected_words_history),
    })


@api.route("/add_char", methods=["POST"])
def api_add_char():
    """Manually add a character to the sentence."""
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
    """Remove the last character."""
    global current_sentence, current_partial_word
    if current_partial_word:
        current_partial_word = current_partial_word[:-1]
    current_sentence = current_sentence[:-1] if current_sentence else ""
    return jsonify({"status": "ok", "sentence": current_sentence})


@api.route("/history", methods=["GET"])
def api_history():
    """Get sentence history."""
    return jsonify({
        "history": [current_sentence] if current_sentence else [],
    })
