"""
Word Formation Engine
Provides intelligent word completion, spell correction, next-word suggestions,
and auto-spacing for building sentences from sign language input.
"""

import os
import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class WordEngine:
    """
    NLP-lite engine for intelligent word formation from sign input.
    
    Features:
    - Word completion from partial input
    - Spell correction (Levenshtein distance)
    - Next-word suggestions (bigram model)
    - Auto-spacing after word detection
    - Common phrase suggestions
    - Misclassification correction (NLP layer)
    """

    # Common sign-detection confusions → correct form
    # Maps misread sequences to intended word
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
        self.bigrams = {}
        self.phrases = []
        self.current_word_buffer = ""

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
            self.bigrams = data.get('bigrams', {})
            self.phrases = data.get('common_phrases', [])
            logger.info(f"WordEngine loaded: {len(self.words)} words, "
                        f"{len(self.bigrams)} bigrams, {len(self.phrases)} phrases")
        except Exception as e:
            logger.error(f"Failed to load dictionary: {e}")

    # ─── Word Completion ─────────────────────────────────────

    def get_completions(self, prefix, max_results=5):
        """
        Get word completions for a prefix.
        
        Args:
            prefix: The partial word typed so far
            max_results: Maximum suggestions to return
        
        Returns:
            List of completion strings
        """
        if not prefix or len(prefix) < 1:
            return []

        prefix = prefix.lower().strip()
        completions = []

        for word in self.words:
            if word.lower().startswith(prefix) and word.lower() != prefix:
                completions.append(word.upper())
                if len(completions) >= max_results:
                    break

        return completions

    # ─── Spell Correction ────────────────────────────────────

    def correct_spelling(self, word, max_distance=2, max_results=3):
        """
        Find closest dictionary words using Levenshtein distance.
        
        Args:
            word: The potentially misspelled word
            max_distance: Maximum edit distance to consider
            max_results: Maximum corrections to return
        
        Returns:
            List of (corrected_word, distance) tuples
        """
        if not word:
            return []

        word_lower = word.lower().strip()

        # Exact match — no correction needed
        if word_lower in self.word_set:
            return []

        candidates = []
        for dict_word in self.words:
            dist = self._levenshtein(word_lower, dict_word.lower())
            if dist <= max_distance:
                candidates.append((dict_word.upper(), dist))

        # Sort by distance, then by word frequency (position in list)
        candidates.sort(key=lambda x: x[1])
        return candidates[:max_results]

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

    # ─── Next-Word Suggestions ───────────────────────────────

    def get_next_words(self, last_word, max_results=5):
        """
        Suggest next words based on bigram frequency.
        
        Args:
            last_word: The last complete word in the sentence
            max_results: Maximum suggestions
        
        Returns:
            List of suggested next words
        """
        if not last_word:
            return []

        key = last_word.lower().strip()
        suggestions = self.bigrams.get(key, [])
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
                # Return just the remaining part
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
        return self.current_word_buffer

    def complete_word(self, word=None):
        """
        Complete the current word and return it.
        Applies spell correction if the word isn't in dictionary.
        
        Args:
            word: Override word (from completion suggestion click)
        
        Returns:
            (final_word, was_corrected, original)
        """
        if word:
            result = word.upper()
            self.current_word_buffer = ""
            return result, False, word

        if not self.current_word_buffer:
            return "", False, ""

        original = self.current_word_buffer
        word_lower = original.lower()

        # Check if it's a real word
        if word_lower in self.word_set:
            self.current_word_buffer = ""
            return original, False, original

        # Try to correct
        corrections = self.correct_spelling(original, max_distance=2, max_results=1)
        if corrections:
            corrected = corrections[0][0]
            self.current_word_buffer = ""
            return corrected, True, original

        # Return as-is
        self.current_word_buffer = ""
        return original, False, original

    def reset_buffer(self):
        """Clear the word buffer."""
        self.current_word_buffer = ""

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

        # Next-word suggestions based on last complete word
        if sentence:
            words = sentence.strip().split()
            if words:
                last = words[-1]
                result['next_words'] = self.get_next_words(last, 5)

        # Phrase completions
        if sentence and len(sentence) >= 3:
            result['phrases'] = self.get_phrase_suggestions(sentence, 3)

        return result
