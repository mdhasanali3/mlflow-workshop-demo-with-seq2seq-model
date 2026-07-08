import unittest

import torch

from src.models.lstm import LSTMDecoder, LSTMEncoder, Seq2SeqLSTM
from src.models.transformer import TransformerSeq2Seq


class ModelSmokeTests(unittest.TestCase):
    def test_lstm_forward_and_backward(self):
        model = Seq2SeqLSTM(
            LSTMEncoder(20, 8, 16, 2, 0.1),
            LSTMDecoder(25, 8, 16, 2, 0.1),
        )
        src = torch.randint(0, 20, (4, 6))
        tgt = torch.randint(0, 25, (4, 7))
        logits = model(src, tgt)[:, 1:]
        loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1, 25), tgt[:, 1:].reshape(-1))
        loss.backward()
        self.assertEqual(logits.shape, (4, 6, 25))

    def test_transformer_forward(self):
        model = TransformerSeq2Seq(
            20, 25, d_model=16, nhead=4, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=32,
        )
        logits = model(torch.randint(0, 20, (4, 6)), torch.randint(0, 25, (4, 5)))
        self.assertEqual(logits.shape, (4, 5, 25))


if __name__ == "__main__":
    unittest.main()
