"""
Prediction Smoothing Module
Implements a robust sliding-window majority-voting system to stabilize
predictions and eliminate flickering. Requires ≥70% agreement before
accepting a prediction, plus configurable stability frames.
"""

import numpy as np
from collections import deque, Counter
import logging
import time

logger = logging.getLogger(__name__)


class PredictionSmoother:
    """
    Smooths predictions using a sliding window with strict majority voting.
    
    A prediction is only accepted when:
      1. It appears in ≥ majority_ratio of the window (default 70%)
      2. Average confidence for that label ≥ confidence_threshold
      3. The label has been stable for ≥ stability_frames consecutive frames
    """

    def __init__(self, window_size=12, confidence_threshold=0.45,
                 majority_ratio=0.70, stability_frames=8):
        """
        Args:
            window_size:          Number of recent predictions to consider
            confidence_threshold: Minimum avg confidence for acceptance
            majority_ratio:       Fraction of window that must agree (0.0–1.0)
            stability_frames:     Consecutive identical predictions required
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.majority_ratio = majority_ratio
        self.stability_frames = stability_frames

        self.prediction_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)

        # Stability tracking
        self._consecutive_label = None
        self._consecutive_count = 0

        # Last accepted prediction (for de-dup)
        self._last_accepted_label = None
        self._last_accepted_time = 0

        logger.info(
            f"PredictionSmoother initialized | "
            f"window={window_size} | threshold={confidence_threshold} | "
            f"majority={majority_ratio:.0%} | stability={stability_frames}"
        )

    def add_prediction(self, label, confidence):
        """
        Add a new frame's prediction to the sliding window.
        
        Args:
            label:      Predicted class label (string)
            confidence: Prediction confidence (0.0–1.0)
        """
        self.prediction_buffer.append(label)
        self.confidence_buffer.append(confidence)

        # Track consecutive identical labels for stability check
        if label == self._consecutive_label:
            self._consecutive_count += 1
        else:
            self._consecutive_label = label
            self._consecutive_count = 1

    def get_smoothed_prediction(self):
        """
        Get the smoothed prediction using strict majority voting.
        
        Returns:
            (label, confidence): Accepted prediction, or (None, 0.0)
            if the window doesn't meet all criteria.
        """
        buf_len = len(self.prediction_buffer)
        if buf_len < 3:
            return None, 0.0

        # Count occurrences of each label in the window
        counter = Counter(self.prediction_buffer)
        majority_label, majority_count = counter.most_common(1)[0]

        # ── Gate 1: Majority ratio ───────────────────────────
        ratio = majority_count / buf_len
        if ratio < self.majority_ratio:
            return None, 0.0

        # ── Gate 2: Average confidence ───────────────────────
        confidences = [
            c for l, c in zip(self.prediction_buffer, self.confidence_buffer)
            if l == majority_label
        ]
        avg_conf = np.mean(confidences)
        if avg_conf < self.confidence_threshold:
            return None, 0.0

        # ── Gate 3: Stability (consecutive frames) ───────────
        if (self._consecutive_label != majority_label or
                self._consecutive_count < self.stability_frames):
            # Allow through if buffer is small (just started)
            if buf_len >= self.stability_frames:
                return None, 0.0

        return majority_label, float(avg_conf)

    def is_new_prediction(self, label):
        """
        Check if this label is different from the last accepted one.
        Used by the caller to avoid re-adding the same letter.
        """
        return label != self._last_accepted_label

    def mark_accepted(self, label):
        """Record that this label was accepted into the sentence."""
        self._last_accepted_label = label
        self._last_accepted_time = time.time()

    def time_since_last_accept(self):
        """Seconds since the last accepted prediction."""
        if self._last_accepted_time == 0:
            return float('inf')
        return time.time() - self._last_accepted_time

    def reset(self):
        """Clear the prediction buffer and all tracking state."""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self._consecutive_label = None
        self._consecutive_count = 0
        self._last_accepted_label = None
        self._last_accepted_time = 0
        logger.info("Prediction buffer cleared")
