"""
Hybrid Prediction Module — Production-Level
Combines static (rule-based) and dynamic (motion-based) gesture recognition.
Automatically switches between modes with mode-lock to prevent jitter.
"""

import os
import sys
import numpy as np
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, LABELS_PATH, SIGN_CLASSES, DATA_DIR, MODE_LOCK_DURATION
from model.hand_rules import SignClassifier, HandFeatures
from model.motion_detector import MotionDetector

logger = logging.getLogger(__name__)

LIBRARY_PATH = os.path.join(DATA_DIR, "sign_library.json")


class GesturePredictor:
    """
    Hybrid gesture predictor combining:
    1. Static classifier (rule-based geometry) — instant, no training needed
    2. Motion detector (trajectory analysis) — for dynamic word signs
    3. Optional Keras model fallback

    Auto-switches between static/dynamic with mode-lock to prevent jitter.
    """

    def __init__(self, model_path=None, labels_path=None):
        self.model = None
        self.labels = SIGN_CLASSES
        self.model_loaded = False

        # Primary: Rule-based static classifier
        self.static_classifier = SignClassifier(library_path=LIBRARY_PATH)

        # Dynamic: Motion detector for word gestures
        self.motion_detector = MotionDetector()

        # Detection mode: 'auto', 'alphabet', 'words'
        self.mode = 'auto'

        # Mode-lock: prevent rapid switching between static/dynamic in auto mode
        self._last_mode_type = 'static'   # 'static' or 'dynamic'
        self._last_mode_switch = 0

        # Last detection info
        self.last_category = None

        # Optional: Load Keras model
        model_path = model_path or MODEL_PATH
        labels_path = labels_path or LABELS_PATH
        self._load_keras_model(model_path, labels_path)

        logger.info(
            f"GesturePredictor initialized | "
            f"Static classifier: ready | "
            f"Motion detector: ready | "
            f"Keras model: {'loaded' if self.model_loaded else 'not available'} | "
            f"Mode: {self.mode}"
        )

    def _load_keras_model(self, model_path, labels_path):
        """Load Keras model as optional enhancement."""
        try:
            if os.path.exists(model_path):
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                import tensorflow as tf
                if hasattr(tf, 'get_logger'):
                    tf.get_logger().setLevel('ERROR')
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                if os.path.exists(labels_path):
                    import json
                    with open(labels_path, "r") as f:
                        self.labels = json.load(f)
                logger.info(f"Keras model loaded as fallback")
        except Exception as e:
            logger.warning(f"Keras model not loaded: {e}")
            self.model_loaded = False

    def predict(self, landmarks):
        """
        Predict gesture using hybrid approach with mode-lock.

        Args:
            landmarks: numpy array of shape (126,) — raw MediaPipe landmarks

        Returns:
            (label, confidence): Best prediction
        """
        if landmarks is None:
            return None, 0.0

        # Get first hand landmarks (21×3)
        hand1 = landmarks[:63].reshape(21, 3)
        has_hand1 = not np.all(hand1 == 0)

        hand2 = None
        if len(landmarks) >= 126:
            hand2_flat = landmarks[63:126]
            if not np.all(hand2_flat == 0):
                hand2 = hand2_flat.reshape(21, 3)

        if not has_hand1 and hand2 is None:
            return None, 0.0

        active_hand = hand1 if has_hand1 else hand2
        features = HandFeatures(active_hand)

        # Update motion detector
        self.motion_detector.update(active_hand)
        is_moving = self.motion_detector.is_hand_moving()

        now = time.time()

        # ── Mode-based prediction ────────────────────────────

        if self.mode == 'alphabet':
            return self._predict_static(landmarks, features)

        if self.mode == 'words':
            # Prefer motion detection for words
            if is_moving:
                label, conf, cat = self.motion_detector.detect_motion(active_hand, features)
                if label:
                    self.last_category = cat
                    return label, conf
            # Fall back to static word signs
            return self._predict_static(landmarks, features, words_priority=True)

        # Auto mode: try both with mode-lock
        static_label, static_conf = self._predict_static(landmarks, features)

        if is_moving:
            # Mode-lock: only switch to dynamic if lock period has elapsed
            can_switch_to_dynamic = (
                self._last_mode_type == 'dynamic' or
                now - self._last_mode_switch > MODE_LOCK_DURATION
            )

            if can_switch_to_dynamic:
                motion_label, motion_conf, motion_cat = self.motion_detector.detect_motion(
                    active_hand, features
                )
                if motion_label and motion_conf > 0.55:
                    if self._last_mode_type != 'dynamic':
                        self._last_mode_type = 'dynamic'
                        self._last_mode_switch = now
                    self.last_category = motion_cat
                    return motion_label, motion_conf

        if static_label:
            if self._last_mode_type != 'static':
                # Mode-lock check for switching back to static
                if now - self._last_mode_switch > MODE_LOCK_DURATION:
                    self._last_mode_type = 'static'
                    self._last_mode_switch = now
                else:
                    return None, 0.0  # Locked in dynamic mode
            return static_label, static_conf

        return None, 0.0

    def _predict_static(self, landmarks, features, words_priority=False):
        """Run static classification on landmarks."""
        # Try first hand
        label, conf, cat = self.static_classifier.classify(landmarks[:63])

        # Try second hand if first didn't match
        if label is None and len(landmarks) >= 126:
            hand2 = landmarks[63:126]
            if not np.all(hand2 == 0):
                label, conf, cat = self.static_classifier.classify(hand2)

        if label:
            self.last_category = cat
            # In words mode, boost word signs over letters
            if words_priority and cat not in ('alphabets', 'numbers'):
                conf = min(1.0, conf * 1.15)

        return label, conf if label else 0.0

    def predict_top_k(self, landmarks, k=3):
        """Get top-K predictions."""
        label, conf = self.predict(landmarks)
        if label:
            return [(label, conf)]
        return []

    def set_mode(self, mode):
        """Set detection mode: 'auto', 'alphabet', 'words'."""
        if mode in ('auto', 'alphabet', 'words'):
            self.mode = mode
            logger.info(f"Detection mode set to: {mode}")
        return self.mode

    def get_library_info(self):
        """Get sign library information."""
        return {
            'signs': self.static_classifier.get_all_signs(),
            'categories': self.static_classifier.get_categories(),
            'total': len(self.static_classifier.get_all_signs()),
        }

    def search_library(self, query):
        """Search the sign library."""
        return self.static_classifier.search_signs(query)

    @property
    def is_ready(self):
        """Always ready — rule-based needs no model file."""
        return True
