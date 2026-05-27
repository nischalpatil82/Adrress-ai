"""
fuzzy_engine.probabilistic
==========================
Peter Norvig's statistical spell correction algorithm.

Provides the ProbabilisticEngine class which:
  1. Builds a frequency dictionary of all valid words in the database.
  2. Generates all possible 1-edit and 2-edit variations of a misspelled word.
  3. Returns the most statistically probable correction based on word frequency.
"""

from collections import Counter
import re


class ProbabilisticEngine:
    """
    Statistical spell correction based on word frequency.
    Generates variations of a word and picks the one that appears
    most often in the known vocabulary.
    """

    def __init__(self, vocab_frequency: Counter):
        """
        Initialize with a pre-built Counter of word frequencies.
        """
        self.freqs = vocab_frequency
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.total_words = sum(self.freqs.values())
        
        self.qwerty = {
            'q': 'wa', 'w': 'qase', 'e': 'wsdf', 'r': 'edfg', 't': 'rfgh', 'y': 'tghj',
            'u': 'yhjki', 'i': 'ujklo', 'o': 'iklp', 'p': 'ol', 'a': 'qwsz', 's': 'awdezx',
            'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn',
            'k': 'jiolm', 'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }

    def probability(self, word: str) -> float:
        """Probability of `word`. P(w) = count(w) / N"""
        if self.total_words == 0:
            return 0.0
        return self.freqs[word] / self.total_words

    def _keyboard_weight(self, original: str, candidate: str) -> float:
        """Applies a penalty multiplier if substituted letters are far apart on QWERTY."""
        weight = 1.0
        if len(original) == len(candidate):
            for c1, c2 in zip(original, candidate):
                if c1 != c2:
                    if c1 in self.qwerty and c2 in self.qwerty[c1]:
                        weight *= 1.2  # Adjacent typo boost
                    else:
                        weight *= 0.2  # Far typo penalty
        return weight

    def _scored_probability(self, word: str, candidate: str) -> float:
        return self.probability(candidate) * self._keyboard_weight(word, candidate)

    def correct(self, word: str) -> str:
        """
        Most probable spelling correction for `word`.
        Follows Norvig's priority:
        1. Known word itself (0 edit)
        2. Known word 1 edit away
        3. Known word 2 edits away
        4. Unknown word (returns itself)
        """
        candidates = (self.known([word]) or 
                      self.known(self.edits1(word)) or 
                      self.known(self.edits2(word)) or 
                      [word])
        
        # Return the candidate with highest keyboard-scored frequency
        return max(candidates, key=lambda c: self._scored_probability(word, c))

    def known(self, words: set) -> set:
        """The subset of `words` that appear in the dictionary."""
        return set(w for w in words if w in self.freqs)

    def edits1(self, word: str) -> set:
        """All strings that are one edit away from `word`."""
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in self.alphabet]
        inserts    = [L + c + R               for L, R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> set:
        """All strings that are two edits away from `word`."""
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
