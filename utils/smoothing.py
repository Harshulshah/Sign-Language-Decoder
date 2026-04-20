"""
Prediction Smoothing Module — Production-Level
Implements hold-to-confirm, cooldown system, detection state machine,
and robust sliding-window majority-voting to stabilize predictions.
"""

import numpy as np
from collections import deque, Counter
import logging
import time

logger = logging.getLogger(__name__)


# Detection states
STATE_IDLE = "idle"              # No hand detected
STATE_DETECTING = "detecting"    # Hand visible, analyzing
STATE_LOCKED = "locked"          # Gesture confirmed (all gates passed)
STATE_FORMING_WORD = "forming"   # Building a word (partial word active)


class PredictionSmoother:
    """
    Smooths predictions using a sliding window with strict majority voting,
    hold-to-confirm mechanism, cooldown system, and detection state machine.

    A prediction is only accepted when:
      1. It appears in ≥ majority_ratio of the window
      2. Average confidence for that label ≥ confidence_threshold
      3. The label has been stable for ≥ stability_frames consecutive frames
      4. The label has been held for ≥ hold_frames consecutive frames (hold-to-confirm)
      5. Cooldown has elapsed since last acceptance of the SAME label
    """

    def __init__(self, window_size=8, confidence_threshold=0.40,
                 majority_ratio=0.60, stability_frames=5,
                 hold_frames=10, cooldown_time=1.5):
        """
        Args:
            window_size:          Number of recent predictions to consider
            confidence_threshold: Minimum avg confidence for acceptance
            majority_ratio:       Fraction of window that must agree (0.0–1.0)
            stability_frames:     Consecutive identical predictions required
            hold_frames:          Frames to hold a gesture before confirming
            cooldown_time:        Seconds before same letter re-accepted
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.majority_ratio = majority_ratio
        self.stability_frames = stability_frames
        self.hold_frames = hold_frames
        self.cooldown_time = cooldown_time

        self.prediction_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)

        # Stability tracking
        self._consecutive_label = None
        self._consecutive_count = 0

        # Hold-to-confirm tracking
        self._hold_label = None
        self._hold_count = 0
        self._hold_confirmed = False

        # Last accepted prediction (for cooldown / de-dup)
        self._last_accepted_label = None
        self._last_accepted_time = 0

        # Cooldown: require gesture change before re-accepting same letter
        self._cooldown_gesture_changed = True

        # Detection state machine
        self._detection_state = STATE_IDLE
        self._state_enter_time = time.time()

        # No-hand tracking
        self._last_hand_time = 0
        self._hand_present = False

        logger.info(
            f"PredictionSmoother initialized | "
            f"window={window_size} | threshold={confidence_threshold} | "
            f"majority={majority_ratio:.0%} | stability={stability_frames} | "
            f"hold={hold_frames} | cooldown={cooldown_time}s"
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
        self._hand_present = True
        self._last_hand_time = time.time()

        # Track consecutive identical labels for stability check
        if label == self._consecutive_label:
            self._consecutive_count += 1
        else:
            self._consecutive_label = label
            self._consecutive_count = 1
            # Gesture changed — allow re-acceptance after cooldown
            if label != self._last_accepted_label:
                self._cooldown_gesture_changed = True

        # Hold-to-confirm tracking
        if label == self._hold_label:
            self._hold_count += 1
        else:
            self._hold_label = label
            self._hold_count = 1
            self._hold_confirmed = False

        # Update state machine
        self._update_state()

    def mark_no_hand(self):
        """Called when no hand is detected in a frame."""
        self._hand_present = False
        if self._detection_state != STATE_IDLE:
            self._set_state(STATE_IDLE)

    def get_smoothed_prediction(self):
        """
        Get the smoothed prediction using strict majority voting
        with hold-to-confirm and cooldown.

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
            if buf_len >= self.stability_frames:
                return None, 0.0

        # ── Gate 4: Hold-to-confirm ──────────────────────────
        if self._hold_label == majority_label:
            if self._hold_count < self.hold_frames:
                # Update state to detecting (working toward confirmation)
                if self._detection_state != STATE_DETECTING:
                    self._set_state(STATE_DETECTING)
                return None, 0.0
            else:
                self._hold_confirmed = True
        else:
            return None, 0.0

        # All gates passed — update state to LOCKED
        if self._detection_state != STATE_LOCKED:
            self._set_state(STATE_LOCKED)

        return majority_label, float(avg_conf)

    def is_new_prediction(self, label):
        """
        Check if this label is different from the last accepted one,
        and that cooldown has elapsed.
        """
        if label != self._last_accepted_label:
            return True
        # Same label — only allow if gesture changed AND cooldown elapsed
        if self._cooldown_gesture_changed and self.time_since_last_accept() > self.cooldown_time:
            return True
        return False

    def is_cooled_down(self, label):
        """
        Check if we can accept this label (cooldown + gesture change).
        Prevents repeated letters unless gesture changes and stabilizes.
        """
        if label != self._last_accepted_label:
            return True
        return (self._cooldown_gesture_changed and
                self.time_since_last_accept() > self.cooldown_time)

    def mark_accepted(self, label):
        """Record that this label was accepted into the sentence."""
        self._last_accepted_label = label
        self._last_accepted_time = time.time()
        self._cooldown_gesture_changed = False
        self._hold_confirmed = False
        self._hold_count = 0

        # Clear the buffer after acceptance to start fresh
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()

    def time_since_last_accept(self):
        """Seconds since the last accepted prediction."""
        if self._last_accepted_time == 0:
            return float('inf')
        return time.time() - self._last_accepted_time

    def time_since_hand(self):
        """Seconds since a hand was last detected."""
        if self._last_hand_time == 0:
            return float('inf')
        return time.time() - self._last_hand_time

    def get_detection_state(self):
        """Return the current detection state string."""
        return self._detection_state

    def get_hold_progress(self):
        """
        Return hold-to-confirm progress as float 0.0–1.0.
        Useful for UI progress indicators.
        """
        if self.hold_frames <= 0:
            return 1.0
        return min(1.0, self._hold_count / self.hold_frames)

    def set_forming_word(self, is_forming):
        """Called by routes to indicate word formation is active."""
        if is_forming and self._detection_state == STATE_LOCKED:
            self._set_state(STATE_FORMING_WORD)
        elif not is_forming and self._detection_state == STATE_FORMING_WORD:
            self._set_state(STATE_LOCKED)

    def _update_state(self):
        """Update detection state based on current tracking."""
        if not self._hand_present:
            self._set_state(STATE_IDLE)
        elif self._hold_confirmed:
            self._set_state(STATE_LOCKED)
        elif self._consecutive_count >= 2:
            if self._detection_state == STATE_IDLE:
                self._set_state(STATE_DETECTING)
        # FORMING_WORD is set externally via set_forming_word()

    def _set_state(self, new_state):
        """Transition to a new state."""
        if new_state != self._detection_state:
            self._detection_state = new_state
            self._state_enter_time = time.time()

    def reset(self):
        """Clear the prediction buffer and all tracking state."""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self._consecutive_label = None
        self._consecutive_count = 0
        self._hold_label = None
        self._hold_count = 0
        self._hold_confirmed = False
        self._last_accepted_label = None
        self._last_accepted_time = 0
        self._cooldown_gesture_changed = True
        self._detection_state = STATE_IDLE
        self._state_enter_time = time.time()
        self._hand_present = False
        logger.info("Prediction buffer cleared")
