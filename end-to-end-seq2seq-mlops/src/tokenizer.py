from collections import Counter
import torch

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sent in sentences:
        counter.update(sent.split())
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def tokenize_sentence(sentence, vocab):
    return [vocab.get(word, vocab['<unk>']) for word in sentence.split()]
