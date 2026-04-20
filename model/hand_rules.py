"""
Enhanced Rule-Based ASL Classifier with Sign Library Support
Recognizes A-Z, 0-9, and static word signs using MediaPipe hand landmark geometry.
Loads sign definitions from the sign library JSON for extensibility.
"""

import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ─── MediaPipe Landmark Indices ──────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# ─── Geometry Helpers ────────────────────────────────────────

def _pt(lm, idx):
    """Get 3D point for a landmark index."""
    return lm[idx]

def _dist(p1, p2):
    """Euclidean distance."""
    return np.linalg.norm(p1 - p2)

def _angle(v1, v2):
    """Angle in degrees between two vectors."""
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

def _finger_extended(lm, tip, dip, pip, mcp):
    """Check if a finger is extended (straight and far from wrist)."""
    tip_pt, dip_pt, pip_pt = _pt(lm, tip), _pt(lm, dip), _pt(lm, pip)
    wrist = _pt(lm, WRIST)
    # Tip further from wrist than PIP
    if _dist(tip_pt, wrist) <= _dist(pip_pt, wrist) * 0.92:
        return False
    # Finger relatively straight
    v1 = tip_pt - dip_pt
    v2 = pip_pt - dip_pt
    return _angle(v1, v2) > 135

def _finger_curled(lm, tip, dip, pip, mcp):
    """Check if finger tip is close to or below MCP (curled into palm)."""
    tip_pt, mcp_pt = _pt(lm, tip), _pt(lm, mcp)
    wrist = _pt(lm, WRIST)
    return _dist(tip_pt, wrist) < _dist(mcp_pt, wrist) * 1.1

def _thumb_extended(lm):
    """Check if thumb is extended outward from the palm."""
    tip = _pt(lm, THUMB_TIP)
    mcp = _pt(lm, THUMB_MCP)
    idx_mcp = _pt(lm, INDEX_MCP)
    return _dist(tip, idx_mcp) > _dist(mcp, idx_mcp) * 0.85

def _thumb_across(lm):
    """Check if thumb is across the palm."""
    tip = _pt(lm, THUMB_TIP)
    mid_mcp = _pt(lm, MIDDLE_MCP)
    ring_mcp = _pt(lm, RING_MCP)
    center = (mid_mcp + ring_mcp) / 2
    return _dist(tip, center) < _dist(mid_mcp, ring_mcp) * 1.6

def _tips_touching(lm, i1, i2, factor=0.55):
    """Check if two landmarks are close together."""
    ref = _dist(_pt(lm, INDEX_MCP), _pt(lm, INDEX_TIP))
    return _dist(_pt(lm, i1), _pt(lm, i2)) < ref * factor

def _pointing_direction(lm, tip_idx, mcp_idx):
    """Return dominant direction of finger: 'up', 'down', 'sideways'."""
    tip = _pt(lm, tip_idx)
    mcp = _pt(lm, mcp_idx)
    dx = abs(tip[0] - mcp[0])
    dy = abs(tip[1] - mcp[1])
    if dy > dx:
        return 'up' if tip[1] < mcp[1] else 'down'
    return 'sideways'


# ─── Feature Extraction ─────────────────────────────────────

class HandFeatures:
    """
    Extract all geometric features from a 21-landmark hand.
    
    Landmarks are normalized to be:
    - Translation-invariant (centered on wrist)
    - Scale-invariant (divided by palm size)
    """

    @staticmethod
    def normalize_landmarks(landmarks_21x3):
        """
        Normalize landmarks to be position- and scale-invariant.
        
        1. Translate so wrist is at origin
        2. Scale so palm_size (wrist→middle_MCP) = 1.0
        
        Returns:
            Normalized landmarks (21, 3) — relative coordinates
        """
        lm = landmarks_21x3.copy()
        wrist = lm[WRIST].copy()
        # Translate: center on wrist
        lm -= wrist
        # Scale: normalize by palm size
        palm_size = np.linalg.norm(lm[MIDDLE_MCP])
        if palm_size > 1e-6:
            lm /= palm_size
        return lm

    def __init__(self, landmarks_21x3):
        # Store both raw and normalized landmarks
        self.lm_raw = landmarks_21x3
        self.lm = self.normalize_landmarks(landmarks_21x3)
        lm = self.lm

        # Finger extension states (use normalized landmarks)
        self.thumb_out = _thumb_extended(lm)
        self.index_up = _finger_extended(lm, INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP)
        self.middle_up = _finger_extended(lm, MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP)
        self.ring_up = _finger_extended(lm, RING_TIP, RING_DIP, RING_PIP, RING_MCP)
        self.pinky_up = _finger_extended(lm, PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP)

        # Finger curled states
        self.index_curled = _finger_curled(lm, INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP)
        self.middle_curled = _finger_curled(lm, MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP)
        self.ring_curled = _finger_curled(lm, RING_TIP, RING_DIP, RING_PIP, RING_MCP)
        self.pinky_curled = _finger_curled(lm, PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP)

        # Derived states
        self.fingers_up_count = sum([self.index_up, self.middle_up, self.ring_up, self.pinky_up])
        self.all_curled = self.index_curled and self.middle_curled and self.ring_curled and self.pinky_curled
        self.thumb_across = _thumb_across(lm)

        # Index direction
        self.index_dir = _pointing_direction(lm, INDEX_TIP, INDEX_MCP) if self.index_up else 'none'

        # Palm size in normalized space should be ~1.0
        self.palm_size = _dist(_pt(lm, WRIST), _pt(lm, MIDDLE_MCP))

        # Finger gaps (normalized)
        if self.index_up and self.middle_up:
            self.index_middle_gap = _dist(_pt(lm, INDEX_TIP), _pt(lm, MIDDLE_TIP)) / (self.palm_size + 1e-8)
        else:
            self.index_middle_gap = 0.0

        # Thumb-index gap for C/O shapes
        self.thumb_index_gap = _dist(_pt(lm, THUMB_TIP), _pt(lm, INDEX_TIP)) / (self.palm_size + 1e-8)

        # ── Advanced features: inter-finger angles ───────────
        tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

        # Angles between adjacent fingers (at the MCP joint region)
        self.finger_angles = []
        for i in range(len(tips) - 1):
            v1 = _pt(lm, tips[i]) - _pt(lm, mcps[i])
            v2 = _pt(lm, tips[i + 1]) - _pt(lm, mcps[i + 1])
            self.finger_angles.append(_angle(v1, v2))

        # Fingertip distance matrix (normalized) — 5×5
        self.tip_distances = {}
        tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, n1 in enumerate(tip_names):
            for j, n2 in enumerate(tip_names):
                if i < j:
                    d = _dist(_pt(lm, tips[i]), _pt(lm, tips[j])) / (self.palm_size + 1e-8)
                    self.tip_distances[f"{n1}_{n2}"] = d


# ─── Sign Classifier ────────────────────────────────────────

class SignClassifier:
    """
    Rule-based ASL sign classifier.
    Matches hand geometry features against known sign patterns.
    """

    def __init__(self, library_path=None):
        self.library = None
        if library_path and os.path.exists(library_path):
            try:
                with open(library_path, 'r', encoding='utf-8') as f:
                    self.library = json.load(f)
                logger.info(f"Sign library loaded: {self.library.get('total_signs', '?')} signs")
            except Exception as e:
                logger.warning(f"Could not load sign library: {e}")

    def classify(self, landmarks_flat):
        """
        Classify a static hand sign from MediaPipe landmarks.

        Args:
            landmarks_flat: numpy array of shape (63,) — one hand (21×3)

        Returns:
            (label, confidence, category): Best match or (None, 0.0, None)
        """
        if landmarks_flat is None or len(landmarks_flat) < 63:
            return None, 0.0, None

        lm = landmarks_flat[:63].reshape(21, 3)
        if np.all(lm == 0):
            return None, 0.0, None

        f = HandFeatures(lm)
        scores = {}

        # ── Alphabets ────────────────────────────────────────
        self._classify_alphabets(f, lm, scores)

        # ── Numbers ──────────────────────────────────────────
        self._classify_numbers(f, lm, scores)

        # ── Static Words ─────────────────────────────────────
        self._classify_static_words(f, lm, scores)

        if not scores:
            return None, 0.0, None

        best = max(scores, key=lambda k: scores[k][0])
        conf, cat = scores[best]
        return best, conf, cat

    def _classify_alphabets(self, f, lm, scores):
        """Classify A-Z alphabets."""

        # A: Fist, thumb beside
        if f.all_curled and f.thumb_out:
            scores['A'] = (0.85, 'alphabets')

        # B: 4 fingers up, thumb across
        if f.index_up and f.middle_up and f.ring_up and f.pinky_up and not f.thumb_out:
            scores['B'] = (0.85, 'alphabets')

        # C: Curved hand
        if not f.all_curled and f.fingers_up_count <= 2:
            if 0.25 < f.thumb_index_gap < 1.0:
                scores['C'] = (0.60, 'alphabets')

        # D: Index up, thumb touches middle
        if f.index_up and f.middle_curled and f.ring_curled and f.pinky_curled:
            if _tips_touching(lm, THUMB_TIP, MIDDLE_TIP, 0.8):
                scores['D'] = (0.80, 'alphabets')

        # E: All curled, thumb tucked
        if f.all_curled and not f.thumb_out:
            scores['E'] = (0.72, 'alphabets')

        # F: Thumb+index touch, 3 fingers up
        if _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.5):
            if f.middle_up and f.ring_up and f.pinky_up:
                scores['F'] = (0.85, 'alphabets')

        # G: Index sideways
        if f.index_up and f.middle_curled and f.ring_curled and f.pinky_curled and f.index_dir == 'sideways':
            scores['G'] = (0.75, 'alphabets')

        # H: Index+middle sideways
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled and f.index_dir == 'sideways':
            scores['H'] = (0.75, 'alphabets')

        # I: Pinky up only
        if f.pinky_up and f.index_curled and f.middle_curled and f.ring_curled:
            scores['I'] = (0.85, 'alphabets')

        # K: Index+middle up, thumb between, spread
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled and f.thumb_out:
            if f.index_middle_gap > 0.3:
                scores['K'] = (0.72, 'alphabets')

        # L: Index up + thumb out at right angle
        if f.index_up and f.middle_curled and f.ring_curled and f.pinky_curled and f.thumb_out:
            if f.index_dir == 'up':
                scores['L'] = (0.87, 'alphabets')

        # M: Fist, 3 fingers over thumb
        if f.all_curled and not f.thumb_out and f.thumb_across:
            scores['M'] = (0.52, 'alphabets')

        # N: Fist, 2 fingers over thumb
        if f.all_curled and not f.thumb_out and f.thumb_across:
            if 'M' not in scores:
                scores['N'] = (0.50, 'alphabets')

        # O: All fingertips touch thumb
        if _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.5):
            if not f.middle_up and not f.ring_up and not f.pinky_up:
                scores['O'] = (0.75, 'alphabets')

        # P: Like K but pointing down
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled:
            if f.index_dir == 'down':
                scores['P'] = (0.70, 'alphabets')

        # Q: Index pointing down
        if f.index_up and f.middle_curled and f.ring_curled and f.pinky_curled:
            if f.index_dir == 'down':
                scores['Q'] = (0.70, 'alphabets')

        # R: Index+middle crossed (close together)
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled:
            if f.index_middle_gap < 0.25:
                scores['R'] = (0.73, 'alphabets')

        # S: Fist with thumb over
        if f.all_curled and not f.thumb_out and f.thumb_across:
            scores['S'] = (0.68, 'alphabets')

        # T: Thumb between index+middle
        if f.all_curled:
            t_tip = _pt(lm, THUMB_TIP)
            idx_pip = _pt(lm, INDEX_PIP)
            mid_pip = _pt(lm, MIDDLE_PIP)
            between = (idx_pip + mid_pip) / 2
            if _dist(t_tip, between) < _dist(idx_pip, mid_pip) * 1.5:
                scores['T'] = (0.58, 'alphabets')

        # U: Index+middle together, pointing up
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled and not f.thumb_out:
            if f.index_dir == 'up' and f.index_middle_gap < 0.35:
                scores['U'] = (0.80, 'alphabets')

        # V: Peace sign — index+middle spread
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled:
            if f.index_dir == 'up' and f.index_middle_gap > 0.35:
                scores['V'] = (0.85, 'alphabets')

        # W: 3 fingers up spread
        if f.index_up and f.middle_up and f.ring_up and f.pinky_curled:
            scores['W'] = (0.80, 'alphabets')

        # X: Index hooked (partially curled, not fully extended or curled)
        if not f.index_up and not f.index_curled and f.middle_curled and f.ring_curled and f.pinky_curled:
            scores['X'] = (0.62, 'alphabets')

        # Y: Thumb + pinky out (hang loose)
        if f.thumb_out and f.pinky_up and f.index_curled and f.middle_curled and f.ring_curled:
            scores['Y'] = (0.90, 'alphabets')

    def _classify_numbers(self, f, lm, scores):
        """Classify 0-9 numbers."""

        # 0: O shape
        if _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.5) and not f.middle_up and not f.ring_up and not f.pinky_up:
            scores['0'] = (0.60, 'numbers')

        # 1: Index up only, pointing up
        if f.index_up and f.middle_curled and f.ring_curled and f.pinky_curled and not f.thumb_out:
            if f.index_dir == 'up':
                scores['1'] = (0.82, 'numbers')

        # 2: V shape (same as V but categorized as number)
        if f.index_up and f.middle_up and f.ring_curled and f.pinky_curled:
            if f.index_dir == 'up' and f.index_middle_gap > 0.35:
                scores['2'] = (0.60, 'numbers')  # Lower than V

        # 3: Thumb + index + middle
        if f.thumb_out and f.index_up and f.middle_up and f.ring_curled and f.pinky_curled:
            scores['3'] = (0.82, 'numbers')

        # 4: 4 fingers up, thumb curled (same shape as B)
        if f.index_up and f.middle_up and f.ring_up and f.pinky_up and not f.thumb_out:
            scores['4'] = (0.60, 'numbers')  # Lower than B

        # 5: All spread open
        if f.thumb_out and f.index_up and f.middle_up and f.ring_up and f.pinky_up:
            scores['5'] = (0.84, 'numbers')

        # 6: Thumb touches pinky, others up
        if _tips_touching(lm, THUMB_TIP, PINKY_TIP, 0.6) and f.index_up and f.middle_up and f.ring_up:
            scores['6'] = (0.78, 'numbers')

        # 7: Thumb touches ring, others up
        if _tips_touching(lm, THUMB_TIP, RING_TIP, 0.6) and f.index_up and f.middle_up and f.pinky_up:
            scores['7'] = (0.78, 'numbers')

        # 8: Thumb touches middle, others up
        if _tips_touching(lm, THUMB_TIP, MIDDLE_TIP, 0.6) and f.index_up and f.ring_up and f.pinky_up:
            scores['8'] = (0.78, 'numbers')

        # 9: Thumb touches index, others up
        if _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.5) and f.middle_up and f.ring_up and f.pinky_up:
            scores['9'] = (0.65, 'numbers')  # Can conflict with F

    def _classify_static_words(self, f, lm, scores):
        """Classify static word signs (no motion required)."""

        # LOVE (ILY): thumb + index + pinky
        if f.thumb_out and f.index_up and not f.middle_up and not f.ring_up and f.pinky_up:
            scores['LOVE'] = (0.92, 'emotions')

        # STOP: Open palm facing forward (all fingers up)
        if f.index_up and f.middle_up and f.ring_up and f.pinky_up:
            if f.thumb_out:
                scores['STOP'] = (0.55, 'actions')  # Lower priority

        # OK: Thumb+index circle, others up
        if _tips_touching(lm, THUMB_TIP, INDEX_TIP, 0.5) and f.middle_up and f.ring_up and f.pinky_up:
            scores['OK'] = (0.82, 'daily_use')

        # PHONE: Y-hand at ear
        if f.thumb_out and f.pinky_up and f.index_curled and f.middle_curled and f.ring_curled:
            scores['PHONE'] = (0.65, 'daily_use')  # Same as Y, context-dependent

    def get_all_signs(self):
        """Return all sign definitions from the library."""
        if not self.library:
            return []

        signs = []
        categories = self.library.get('categories', {})
        for cat_key, cat_data in categories.items():
            for sign_label, sign_info in cat_data.get('signs', {}).items():
                signs.append({
                    'label': sign_label,
                    'category': cat_key,
                    'category_label': cat_data.get('label', cat_key),
                    'category_icon': cat_data.get('icon', ''),
                    'type': sign_info.get('type', 'static'),
                    'description': sign_info.get('description', ''),
                })
        return signs

    def search_signs(self, query):
        """Search signs by label or description."""
        query = query.lower().strip()
        results = []
        for sign in self.get_all_signs():
            if (query in sign['label'].lower() or
                    query in sign['description'].lower() or
                    query in sign['category'].lower()):
                results.append(sign)
        return results

    def get_categories(self):
        """Return list of categories with counts."""
        if not self.library:
            return []
        cats = []
        for key, data in self.library.get('categories', {}).items():
            cats.append({
                'key': key,
                'label': data.get('label', key),
                'icon': data.get('icon', ''),
                'count': len(data.get('signs', {})),
            })
        return cats
