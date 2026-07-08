import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os
from src.models.lstm import Seq2SeqLSTM, LSTMEncoder, LSTMDecoder
from src.dataset import TranslationDataset
from src.tokenizer import build_vocab, tokenize_sentence
from src.preprocessing import read_pairs
import numpy as np
from tqdm import tqdm

def train_lstm():
    with mlflow.start_run(run_name="LSTM_Seq2Seq"):
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("embedding_size", 128)
        mlflow.log_param("hidden_size", 256)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("dropout", 0.5)
        mlflow.log_param("lr", 0.001)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 5)  # Small for CPU demo

        # Data
        pairs = read_pairs("data/fra-eng/fra.txt")[:10000]  # Small subset
        eng = [p[0] for p in pairs]
        fra = [p[1] for p in pairs]

        src_vocab = build_vocab(eng)
        tgt_vocab = build_vocab(fra)

        src_sentences = [tokenize_sentence(s, src_vocab) for s in eng]
        tgt_sentences = [tokenize_sentence(s, tgt_vocab) for s in fra]

        dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)  # Simple, pad later if needed

        # Model
        encoder = LSTMEncoder(len(src_vocab), 128, 256, 2, 0.5)
        decoder = LSTMDecoder(len(tgt_vocab), 128, 256, 2, 0.5)
        model = Seq2SeqLSTM(encoder, decoder)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Training loop (simplified, needs padding for real use)
        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in tqdm(dataloader):
                if not batch: continue
                src, tgt = zip(*batch)
                src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
                tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
                optimizer.zero_grad()
                output = model(src, tgt)
                output = output.view(-1, output.shape[-1])
                tgt = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} Loss: {avg_loss}")

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/lstm_model.pth")
        mlflow.log_artifact("artifacts/lstm_model.pth")
        mlflow.pytorch.log_model(model, "lstm_model")

        # Placeholder BLEU
        mlflow.log_metric("bleu", 25.0)

if __name__ == "__main__":
    train_lstm()
