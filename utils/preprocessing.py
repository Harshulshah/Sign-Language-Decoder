"""
Preprocessing Module
Handles feature extraction and data normalization for the ML model.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks relative to the wrist position
    and scale to unit range for model-agnostic input.
    
    Args:
        landmarks: numpy array of shape (126,) — raw MediaPipe landmarks
        
    Returns:
        normalized: numpy array of shape (126,) — normalized landmarks
    """
    if landmarks is None:
        return None

    landmarks = landmarks.copy()

    # Process each hand separately (63 values each = 21 landmarks × 3)
    for hand_offset in [0, 63]:
        hand = landmarks[hand_offset:hand_offset + 63]

        # Check if this hand has any data
        if np.all(hand == 0):
            continue

        # Extract reference point (wrist = landmark 0)
        wrist_x = hand[0]
        wrist_y = hand[1]
        wrist_z = hand[2]

        # Center on wrist
        for i in range(0, 63, 3):
            hand[i] -= wrist_x
            hand[i + 1] -= wrist_y
            hand[i + 2] -= wrist_z

        # Scale to [-1, 1] range
        max_val = np.max(np.abs(hand))
        if max_val > 0:
            hand /= max_val

        landmarks[hand_offset:hand_offset + 63] = hand

    return landmarks


def compute_finger_angles(landmarks):
    """
    Compute angles between finger joints for richer feature representation.
    
    Args:
        landmarks: numpy array of shape (126,)
        
    Returns:
        angles: numpy array of joint angles
    """
    if landmarks is None:
        return None

    angles = []

    # Finger tip indices for one hand (MediaPipe)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mids = [3, 7, 11, 15, 19]
    finger_bases = [2, 6, 10, 14, 18]

    for hand_offset in [0, 63]:
        hand = landmarks[hand_offset:hand_offset + 63].reshape(21, 3)

        if np.all(hand == 0):
            angles.extend([0.0] * 5)
            continue

        for tip, mid, base in zip(finger_tips, finger_mids, finger_bases):
            v1 = hand[tip] - hand[mid]
            v2 = hand[base] - hand[mid]

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)

    return np.array(angles, dtype=np.float32)


def extract_features(landmarks):
    """
    Combine normalized landmarks with computed angles for a richer feature set.
    
    Args:
        landmarks: numpy array of shape (126,) — raw landmarks
        
    Returns:
        features: numpy array — combined feature vector
    """
    if landmarks is None:
        return None

    normalized = normalize_landmarks(landmarks)
    angles = compute_finger_angles(landmarks)

    # Combine: 126 landmark coords + 10 finger angles = 136 features
    features = np.concatenate([normalized, angles])
    return features


FEATURE_DIM = 136  # 126 landmarks + 10 finger angles
