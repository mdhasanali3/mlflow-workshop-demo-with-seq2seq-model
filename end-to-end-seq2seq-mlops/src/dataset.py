import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = [self.src_vocab['<sos>']] + self.src_sentences[idx] + [self.src_vocab['<eos>']]
        tgt = [self.tgt_vocab['<sos>']] + self.tgt_sentences[idx] + [self.tgt_vocab['<eos>']]
        return torch.tensor(src), torch.tensor(tgt)
