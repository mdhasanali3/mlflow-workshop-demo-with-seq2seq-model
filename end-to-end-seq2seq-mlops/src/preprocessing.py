import re
import unicodedata

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_pairs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pairs = [[normalize_string(s) for s in l.strip().split('\t')[:2]] for l in lines]
    return pairs
