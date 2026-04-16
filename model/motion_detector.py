"""
Motion Detector Module
Tracks hand trajectory over time to detect dynamic gestures.
Uses motion patterns (wave, circle, nod, push) to recognize word signs
that require movement rather than static poses.
"""

import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

# Motion pattern constants
MOTION_BUFFER_SIZE = 30   # Frames of position history
MOTION_THRESHOLD = 0.015  # Minimum movement to count as motion
STILLNESS_THRESHOLD = 0.008  # Below this = hand is still


class MotionDetector:
    """
    Tracks hand center position over time and detects motion patterns
    that map to dynamic sign language gestures.
    """

    def __init__(self, buffer_size=MOTION_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.position_history = deque(maxlen=buffer_size)
        self.time_history = deque(maxlen=buffer_size)
        self.last_detection_time = 0
        self.cooldown = 2.0  # Seconds between dynamic detections
        logger.info(f"MotionDetector initialized | buffer={buffer_size}")

    def update(self, landmarks_21x3):
        """
        Add current hand position to history.
        
        Args:
            landmarks_21x3: numpy array shape (21, 3)
        """
        if landmarks_21x3 is None:
            return

        # Use palm center (average of wrist + MCP joints) as tracking point
        palm_points = landmarks_21x3[[0, 5, 9, 13, 17]]
        center = np.mean(palm_points, axis=0)

        self.position_history.append(center)
        self.time_history.append(time.time())

    def detect_motion(self, landmarks_21x3, hand_features=None):
        """
        Analyze motion history and detect dynamic gesture patterns.
        
        Args:
            landmarks_21x3: current frame landmarks
            hand_features: HandFeatures object from static classifier
        
        Returns:
            (label, confidence, category) or (None, 0.0, None)
        """
        if len(self.position_history) < 10:
            return None, 0.0, None

        now = time.time()
        if now - self.last_detection_time < self.cooldown:
            return None, 0.0, None

        positions = np.array(list(self.position_history))
        n = len(positions)

        # Calculate total displacement and speed
        displacements = np.diff(positions, axis=0)
        distances = np.linalg.norm(displacements[:, :2], axis=1)  # XY only
        total_distance = np.sum(distances)
        
        if total_distance < MOTION_THRESHOLD * 5:
            return None, 0.0, None  # Not enough motion

        # Analyze motion pattern
        result = self._analyze_pattern(positions, displacements, distances, hand_features)
        
        if result[0] is not None:
            self.last_detection_time = now
            self.position_history.clear()
            logger.debug(f"Motion detected: {result[0]} ({result[1]:.0%})")
        
        return result

    def _analyze_pattern(self, positions, displacements, distances, features):
        """
        Determine motion pattern type from position history.
        
        Returns:
            (label, confidence, category)
        """
        n = len(positions)
        
        # XY components
        dx = displacements[:, 0]  # Horizontal
        dy = displacements[:, 1]  # Vertical
        
        # Direction changes
        dx_signs = np.sign(dx)
        dy_signs = np.sign(dy)
        dx_changes = np.sum(np.abs(np.diff(dx_signs)) > 0)
        dy_changes = np.sum(np.abs(np.diff(dy_signs)) > 0)
        
        # Net displacement
        net_x = positions[-1][0] - positions[0][0]
        net_y = positions[-1][1] - positions[0][1]
        total_dist = np.sum(distances)
        
        # Dominant direction
        total_dx = np.sum(np.abs(dx))
        total_dy = np.sum(np.abs(dy))
        horizontal_dominant = total_dx > total_dy * 1.3
        vertical_dominant = total_dy > total_dx * 1.3

        scores = {}

        # ── WAVE (HELLO/HI/BYE): Side-to-side oscillation ───
        if horizontal_dominant and dx_changes >= 3:
            if features and features.fingers_up_count >= 3:
                scores['HELLO'] = (0.80, 'greetings')

        # ── NOD (YES): Vertical oscillation with fist ────────
        if vertical_dominant and dy_changes >= 2:
            if features and features.all_curled:
                scores['YES'] = (0.75, 'daily_use')

        # ── SHAKE (NO): Horizontal shake with specific hand shape
        if horizontal_dominant and dx_changes >= 2:
            if features and features.index_up and features.middle_up:
                scores['NO'] = (0.70, 'daily_use')

        # ── PUSH FORWARD (THANK YOU): Forward motion from body
        if abs(net_y) > 0.03 and net_y > 0:  # Moving down/forward
            if features and features.fingers_up_count >= 3 and not features.all_curled:
                scores['THANK YOU'] = (0.65, 'polite_words')

        # ── CIRCLE ON CHEST (PLEASE/SORRY): Circular motion
        if dx_changes >= 2 and dy_changes >= 2:
            # Check for circular pattern
            half = n // 2
            first_half_dx = np.mean(dx[:half])
            second_half_dx = np.mean(dx[half:])
            if np.sign(first_half_dx) != np.sign(second_half_dx):
                if features and features.fingers_up_count >= 3:
                    scores['PLEASE'] = (0.60, 'polite_words')
                elif features and features.all_curled:
                    scores['SORRY'] = (0.60, 'polite_words')

        # ── RISE (HELP): Upward motion
        if vertical_dominant and net_y < -0.04:  # Moving up (y decreases)
            if features and features.all_curled:
                scores['HELP'] = (0.65, 'actions')
            elif features and features.fingers_up_count >= 4:
                scores['HELP ME'] = (0.60, 'emergency')

        # ── POINT FORWARD (GO/COME)
        if not horizontal_dominant and not vertical_dominant:
            if total_dist > 0.05:
                if net_y > 0.02:  # Away
                    scores['GO'] = (0.55, 'actions')
                elif net_y < -0.02:  # Toward
                    scores['COME'] = (0.55, 'actions')

        # ── DOWNWARD MOTION (GOOD/BAD)
        if vertical_dominant and net_y > 0.03:
            if features and features.fingers_up_count >= 3:
                scores['GOOD'] = (0.55, 'daily_use')

        # ── WIGGLE (WAIT)
        if dx_changes >= 4 and dy_changes >= 4:
            if features and features.fingers_up_count >= 3:
                scores['WAIT'] = (0.55, 'actions')

        if not scores:
            return None, 0.0, None

        best = max(scores, key=lambda k: scores[k][0])
        return best, scores[best][0], scores[best][1]

    def is_hand_moving(self):
        """Check if the hand is currently in motion."""
        if len(self.position_history) < 5:
            return False
        
        recent = list(self.position_history)[-5:]
        positions = np.array(recent)
        displacements = np.diff(positions[:, :2], axis=0)
        avg_speed = np.mean(np.linalg.norm(displacements, axis=1))
        return avg_speed > STILLNESS_THRESHOLD

    def reset(self):
        """Clear motion history."""
        self.position_history.clear()
        self.time_history.clear()
