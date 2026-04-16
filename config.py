"""
Configuration file for the Sign Language Decoder system.
Contains all tunable parameters and settings.
"""

import os

# ─── Base Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ─── Model Settings ─────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ─── Sign Library ────────────────────────────────────────────
SIGN_LIBRARY_PATH = os.path.join(DATA_DIR, "sign_library.json")
CUSTOM_SIGNS_PATH = os.path.join(DATA_DIR, "custom_signs.json")
WORD_DICTIONARY_PATH = os.path.join(DATA_DIR, "word_dictionary.json")

# Classes: A-Z + 0-9 + common words
SIGN_CLASSES = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("0123456789") +
    ["HELLO", "THANKS", "YES", "NO", "PLEASE", "SORRY", "HELP", "LOVE", "SPACE", "DELETE"]
)
NUM_CLASSES = len(SIGN_CLASSES)

# ─── MediaPipe Settings ─────────────────────────────────────
MEDIAPIPE_MAX_HANDS = 2
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# ─── Camera Settings ────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# ─── Prediction Settings ────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45          # Raised: only accept high-confidence
SMOOTHING_WINDOW = 12                # Frames for majority voting (was 3)
MAJORITY_VOTE_RATIO = 0.70           # ≥70% of window must agree
PREDICTION_COOLDOWN = 1.2            # Seconds between adding characters
STABILITY_REQUIRED_FRAMES = 8        # Min identical frames before acceptance
DEFAULT_DETECTION_MODE = "auto"      # 'auto', 'alphabet', 'words'

# ─── Word Engine Settings ───────────────────────────────────
MAX_SUGGESTIONS = 5            # Max word completion suggestions
CORRECTION_THRESHOLD = 2       # Max Levenshtein distance for corrections
AUTO_SPACE_DELAY = 2.0         # Seconds of no new letter before auto-spacing
WORD_SIGN_AUTO_SPACE = True    # Automatically add space after word signs

# ─── Motion Detection Settings ──────────────────────────────
MOTION_BUFFER_FRAMES = 30     # Frames of position history  
MOTION_COOLDOWN = 2.0         # Seconds between dynamic detections

# ─── Performance Settings ───────────────────────────────────
PREDICTION_SKIP_FRAMES = 3    # Process every 3rd frame (was 1)
MAX_PREDICTION_FPS = 15       # Cap prediction rate

# ─── TTS Settings ───────────────────────────────────────────
TTS_RATE = 150                # Speech rate (words per minute)
TTS_VOLUME = 0.9              # Volume 0.0–1.0
TTS_MIN_WORD_LENGTH = 1       # Minimum word length to speak
TTS_COOLDOWN = 1.5            # Minimum seconds between TTS outputs
TTS_SPEAK_PARTIAL = False     # Never speak partial/incomplete words

# ─── Flask Settings ─────────────────────────────────────────
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# ─── Logging ─────────────────────────────────────────────────
LOG_FILE = os.path.join(BASE_DIR, "app.log")
LOG_LEVEL = "DEBUG"
