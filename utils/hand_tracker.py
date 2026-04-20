"""
Hand Tracker Module — Production-Level
Uses MediaPipe Hands to detect and track hand landmarks in real-time.
Includes resolution optimization for faster processing.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

from config import PROCESS_RESOLUTION_W, PROCESS_RESOLUTION_H

logger = logging.getLogger(__name__)


class HandTracker:
    """
    Real-time hand detection and landmark extraction using MediaPipe.
    Supports single and dual-hand detection with configurable confidence.
    Includes performance optimizations: resolution downscaling and frame caching.
    """

    def __init__(self, max_hands=2, min_detection_conf=0.4, min_tracking_conf=0.4):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

        # Cache for last results (performance optimization)
        self._last_results = None
        self._frame_count = 0

        # Resolution optimization
        self._process_w = PROCESS_RESOLUTION_W
        self._process_h = PROCESS_RESOLUTION_H

        logger.info(
            f"HandTracker initialized | max_hands={max_hands} | "
            f"det_conf={min_detection_conf} | track_conf={min_tracking_conf} | "
            f"process_res={self._process_w}x{self._process_h}"
        )

    def find_hands(self, frame, draw=True):
        """
        Detect hands in a frame and optionally draw landmarks.
        Downscales for MediaPipe processing, landmarks are normalized anyway.

        Args:
            frame: BGR image (numpy array)
            draw: Whether to draw landmarks on the frame

        Returns:
            frame: Annotated frame
            results: MediaPipe hand detection results
        """
        # Downscale for processing (landmarks are normalized 0-1)
        h, w = frame.shape[:2]
        if w > self._process_w or h > self._process_h:
            process_frame = cv2.resize(frame, (self._process_w, self._process_h))
        else:
            process_frame = frame

        rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        # Optimization: set writeable flag
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        self._last_results = results
        self._frame_count += 1

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame, results

    def extract_landmarks(self, results):
        """
        Extract normalized landmark coordinates from detection results.
        Returns a flat array of [x, y, z] for each of the 21 keypoints.

        Args:
            results: MediaPipe hand detection results

        Returns:
            landmarks: numpy array of shape (126,) for two hands.
                       Returns None if no hands detected.
        """
        if not results.multi_hand_landmarks:
            return None

        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.extend([lm.x, lm.y, lm.z])
            all_landmarks.extend(hand_data)

        # Pad to 126 values (2 hands × 21 landmarks × 3 coords)
        while len(all_landmarks) < 126:
            all_landmarks.extend([0.0] * 63)

        return np.array(all_landmarks[:126], dtype=np.float32)

    def get_hand_bbox(self, frame, results):
        """
        Get bounding boxes for detected hands.

        Args:
            frame: BGR image
            results: MediaPipe detection results

        Returns:
            List of (x, y, w, h) bounding boxes
        """
        bboxes = []
        if not results.multi_hand_landmarks:
            return bboxes

        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            bboxes.append((
                max(0, x_min), max(0, y_min),
                min(w, x_max) - max(0, x_min),
                min(h, y_max) - max(0, y_min),
            ))
        return bboxes

    def get_hand_count(self, results):
        """Return number of detected hands."""
        if results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
        logger.info("HandTracker resources released")
