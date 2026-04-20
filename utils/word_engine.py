"""
Word Formation Engine — Production-Level
Provides intelligent word completion, spell correction, next-word suggestions
using bigram + trigram models, frequency-weighted scoring, and smart word buffer.
"""

import os
import json
import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)


class WordBuffer:
    """
    Smart word buffer that manages letter → word → sentence pipeline.
    Auto-resets after configurable pause.
    """

    def __init__(self, auto_reset_delay=3.0):
        self.letters = []
        self.last_letter_time = 0
        self.auto_reset_delay = auto_reset_delay

    def add_letter(self, letter):
        """Add a letter, auto-reset if too much time has passed."""
        now = time.time()
        if (self.last_letter_time > 0 and
                now - self.last_letter_time > self.auto_reset_delay):
            self.letters.clear()
        self.letters.append(letter.upper())
        self.last_letter_time = now

    def get_word(self):
        """Get current word from buffer."""
        return "".join(self.letters)

    def backspace(self):
        """Remove last letter."""
        if self.letters:
            self.letters.pop()

    def clear(self):
        """Clear the buffer."""
        self.letters.clear()
        self.last_letter_time = 0

    def is_stale(self):
        """Check if buffer has been idle too long."""
        if not self.letters or self.last_letter_time == 0:
            return False
        return time.time() - self.last_letter_time > self.auto_reset_delay


class WordEngine:
    """
    NLP-lite engine for intelligent word formation from sign input.

    Features:
    - Word completion from partial input (frequency-weighted)
    - Spell correction (Levenshtein distance)
    - Next-word suggestions (bigram + trigram model)
    - Auto-spacing after word detection
    - Common phrase suggestions
    - Misclassification correction (NLP layer)
    - Smart word buffer with auto-reset
    """

    # Common sign-detection confusions → correct form
    MISCLASS_TABLE = {
        # Transpositions
        "HLELO": "HELLO", "HELO": "HELLO", "HLLEO": "HELLO",
        "HLLO": "HELLO", "HELOO": "HELLO", "EHLLO": "HELLO",
        "THNK": "THANK", "THNAKS": "THANKS", "TAHK": "THANK",
        "THNK YOU": "THANK YOU",
        "PLASE": "PLEASE", "PLEES": "PLEASE", "PLESAE": "PLEASE",
        "SRORY": "SORRY", "SORY": "SORRY", "SORR": "SORRY",
        "HLEP": "HELP", "HEPL": "HELP", "HALP": "HELP",
        "YEES": "YES", "YSE": "YES",
        "NOO": "NO",
        "GOOOD": "GOOD", "GOD": "GOOD",
        "WATR": "WATER", "WTER": "WATER",
        "FOOOD": "FOOD", "FODO": "FOOD",
        "HPPY": "HAPPY", "HAPY": "HAPPY",
        "LVOE": "LOVE", "LOEV": "LOVE",
        # Common double-letter errors from sticky detection
        "HELLLO": "HELLO", "HELLP": "HELP",
        "THAANK": "THANK", "PLEAASE": "PLEASE",
        "SOORRY": "SORRY", "GOOOD": "GOOD",
    }

    def __init__(self, dictionary_path=None):
        self.words = []
        self.word_set = set()
        self.word_freq = {}  # word → frequency rank (lower = more common)
        self.bigrams = {}
        self.trigrams = {}
        self.phrases = []
        self.current_word_buffer = ""
        self.smart_buffer = WordBuffer()

        if dictionary_path and os.path.exists(dictionary_path):
            self._load_dictionary(dictionary_path)
        else:
            logger.warning("No dictionary file — word suggestions disabled")

    def _load_dictionary(self, path):
        """Load the word dictionary JSON."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.words = data.get('words_by_frequency', [])
            self.word_set = set(w.lower() for w in self.words)
            # Build frequency map (position = rank)
            for i, w in enumerate(self.words):
                self.word_freq[w.lower()] = i
            self.bigrams = data.get('bigrams', {})
            self.trigrams = data.get('trigrams', {})
            self.phrases = data.get('common_phrases', [])
            logger.info(f"WordEngine loaded: {len(self.words)} words, "
                        f"{len(self.bigrams)} bigrams, "
                        f"{len(self.trigrams)} trigrams, "
                        f"{len(self.phrases)} phrases")
        except Exception as e:
            logger.error(f"Failed to load dictionary: {e}")

    # ─── Word Completion (Frequency-Weighted) ────────────────

    def get_completions(self, prefix, max_results=5):
        """
        Get word completions for a prefix, ranked by frequency.

        Args:
            prefix: The partial word typed so far
            max_results: Maximum suggestions to return

        Returns:
            List of completion strings
        """
        if not prefix or len(prefix) < 1:
            return []

        prefix = prefix.lower().strip()
        candidates = []

        for word in self.words:
            if word.lower().startswith(prefix) and word.lower() != prefix:
                freq_rank = self.word_freq.get(word.lower(), len(self.words))
                candidates.append((word.upper(), freq_rank))

        # Sort by frequency rank (lower = more common = better)
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates[:max_results]]

    # ─── Spell Correction ────────────────────────────────────

    def correct_spelling(self, word, max_distance=2, max_results=3):
        """
        Find closest dictionary words using Levenshtein distance.
        First checks misclassification table for known sign-detection errors.

        Args:
            word: The potentially misspelled word
            max_distance: Maximum edit distance to consider
            max_results: Maximum corrections to return

        Returns:
            List of (corrected_word, distance) tuples
        """
        if not word:
            return []

        word_upper = word.upper().strip()
        word_lower = word.lower().strip()

        # Check misclassification table first
        if word_upper in self.MISCLASS_TABLE:
            corrected = self.MISCLASS_TABLE[word_upper]
            return [(corrected, 0)]

        # Exact match — no correction needed
        if word_lower in self.word_set:
            return []

        candidates = []
        for dict_word in self.words:
            dist = self._levenshtein(word_lower, dict_word.lower())
            if dist <= max_distance:
                freq_rank = self.word_freq.get(dict_word.lower(), len(self.words))
                candidates.append((dict_word.upper(), dist, freq_rank))

        # Sort by distance first, then by frequency rank
        candidates.sort(key=lambda x: (x[1], x[2]))
        return [(c[0], c[1]) for c in candidates[:max_results]]

    @staticmethod
    def _levenshtein(s1, s2):
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return WordEngine._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    # ─── Next-Word Suggestions (Bigram + Trigram) ────────────

    def get_next_words(self, sentence_words, max_results=5):
        """
        Suggest next words based on trigram then bigram frequency.

        Args:
            sentence_words: List of last words in the sentence
            max_results: Maximum suggestions

        Returns:
            List of suggested next words
        """
        if not sentence_words:
            return []

        # Try trigram first (last 2 words)
        if len(sentence_words) >= 2 and self.trigrams:
            key = f"{sentence_words[-2].lower()} {sentence_words[-1].lower()}"
            suggestions = self.trigrams.get(key, [])
            if suggestions:
                return [w.upper() for w in suggestions[:max_results]]

        # Fall back to bigram (last word)
        last_word = sentence_words[-1].lower().strip()
        suggestions = self.bigrams.get(last_word, [])
        return [w.upper() for w in suggestions[:max_results]]

    # ─── Phrase Suggestions ──────────────────────────────────

    def get_phrase_suggestions(self, current_text, max_results=3):
        """
        Suggest complete phrases based on current sentence start.

        Args:
            current_text: Current sentence text
            max_results: Maximum phrases to return

        Returns:
            List of matching phrase completions
        """
        if not current_text or len(current_text) < 2:
            return []

        text_lower = current_text.lower().strip()
        matches = []

        for phrase in self.phrases:
            if phrase.startswith(text_lower) and phrase != text_lower:
                remaining = phrase[len(text_lower):].strip()
                if remaining:
                    matches.append(remaining.upper())
                if len(matches) >= max_results:
                    break

        return matches

    # ─── Word Buffer Management ──────────────────────────────

    def add_letter(self, letter):
        """Add a letter to the current word buffer."""
        self.current_word_buffer += letter.upper()
        self.smart_buffer.add_letter(letter)
        return self.current_word_buffer

    def complete_word(self, word=None):
        """
        Complete the current word and return it.
        Checks misclassification table first, then applies spell correction.

        Args:
            word: Override word (from completion suggestion click)

        Returns:
            (final_word, was_corrected, original)
        """
        if word:
            result = word.upper()
            self.current_word_buffer = ""
            self.smart_buffer.clear()
            return result, False, word

        if not self.current_word_buffer:
            return "", False, ""

        original = self.current_word_buffer
        word_upper = original.upper()
        word_lower = original.lower()

        # 1. Check misclassification table
        if word_upper in self.MISCLASS_TABLE:
            corrected = self.MISCLASS_TABLE[word_upper]
            self.current_word_buffer = ""
            self.smart_buffer.clear()
            return corrected, True, original

        # 2. Check if it's a real word
        if word_lower in self.word_set:
            self.current_word_buffer = ""
            self.smart_buffer.clear()
            return original, False, original

        # 3. Try to correct via Levenshtein
        corrections = self.correct_spelling(original, max_distance=2, max_results=1)
        if corrections:
            corrected = corrections[0][0]
            self.current_word_buffer = ""
            self.smart_buffer.clear()
            return corrected, True, original

        # Return as-is
        self.current_word_buffer = ""
        self.smart_buffer.clear()
        return original, False, original

    def reset_buffer(self):
        """Clear the word buffer."""
        self.current_word_buffer = ""
        self.smart_buffer.clear()

    def get_suggestions(self, sentence, current_partial=""):
        """
        Get all suggestions for the current state.

        Returns dict with:
        - completions: word completions for current partial
        - next_words: next word suggestions
        - phrases: phrase completions
        - corrections: spell corrections for current partial
        """
        result = {
            'completions': [],
            'next_words': [],
            'phrases': [],
            'corrections': [],
        }

        # Word completions
        if current_partial and len(current_partial) >= 1:
            result['completions'] = self.get_completions(current_partial, 5)
            if len(current_partial) >= 2:
                result['corrections'] = [
                    c[0] for c in self.correct_spelling(current_partial, 2, 3)
                ]

        # Next-word suggestions based on last complete words
        if sentence:
            words = sentence.strip().split()
            if words:
                result['next_words'] = self.get_next_words(words, 5)

        # Phrase completions
        if sentence and len(sentence) >= 3:
            result['phrases'] = self.get_phrase_suggestions(sentence, 3)

        return result
