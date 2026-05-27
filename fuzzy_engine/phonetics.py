"""
fuzzy_engine.phonetics
======================
Bespoke phonetic hashing module tailored to Indian address pronunciation.
Handles common Indian dialect swaps:
  - 'v' <-> 'w'
  - 'i' <-> 'ee' <-> 'y'
  - 'u' <-> 'oo'
  - 'sh' <-> 's'
  - 'ph' <-> 'f'
"""
import re

def indian_phonetic_hash(word: str) -> str:
    """
    Creates a phonetic signature of a word, highly tolerant of
    Indian spelling variations for the same sound.
    """
    if not word or len(word) < 2:
        return word
        
    w = word.lower()
    
    # 1. Normalize common digraphs and equivalents
    w = w.replace('ph', 'f')
    w = w.replace('sh', 's')
    w = w.replace('sz', 's')
    w = w.replace('ch', 'c')
    w = w.replace('gh', 'g')
    w = w.replace('kh', 'k')
    w = w.replace('jh', 'j')
    w = w.replace('th', 't')
    w = w.replace('dh', 'd')
    w = w.replace('bh', 'b')
    w = w.replace('x', 'ks')
    w = w.replace('z', 'j')
    w = w.replace('q', 'k')

    # 2. Normalize vowel stretches
    w = re.sub(r'ee+', 'i', w)
    w = re.sub(r'oo+', 'u', w)
    w = re.sub(r'aa+', 'a', w)
    
    # 3. Deal with 'y' sounding like 'i' at ends or between consonants
    w = w.replace('y', 'i')
    
    # 4. v/w swap
    w = w.replace('v', 'w')
    
    # 5. Remove adjacent duplicates (e.g., 'nn' -> 'n')
    w = re.sub(r'(.)\1+', r'\1', w)
    
    # 6. Keep first letter intact, remove all other vowels to create a harsh skeleton
    # Vowels in Indian names are often dropped or mangled.
    if len(w) > 1:
        first_letter = w[0]
        rest = re.sub(r'[aeiou]', '', w[1:])
        w = first_letter + rest
        
    return w

class PhoneticEngine:
    """Maps phonetic hashes back to valid DB words."""
    def __init__(self, vocab_words: list):
        # Build map: hash -> list of actual words
        self.hash_map = {}
        for w in vocab_words:
            h = indian_phonetic_hash(w)
            if h not in self.hash_map:
                self.hash_map[h] = []
            self.hash_map[h].append(w)
            
    def get_matches(self, word: str) -> list:
        """Return words with the same exact phonetic hash."""
        h = indian_phonetic_hash(word)
        return self.hash_map.get(h, [])
