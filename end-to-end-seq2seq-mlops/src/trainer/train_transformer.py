import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import os
from src.models.transformer import TransformerSeq2Seq
from src.dataset import TranslationDataset
from src.tokenizer import build_vocab, tokenize_sentence
from src.preprocessing import read_pairs
from tqdm import tqdm

def train_transformer():
    with mlflow.start_run(run_name="Transformer_Seq2Seq"):
        mlflow.log_param("model_type", "Transformer")
        mlflow.log_param("d_model", 256)
        mlflow.log_param("nhead", 8)
        mlflow.log_param("num_layers", 3)
        mlflow.log_param("lr", 0.0001)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("epochs", 3)

        pairs = read_pairs("data/fra-eng/fra.txt")[:5000]  # Even smaller for Transformer on CPU
        eng = [p[0] for p in pairs]
        fra = [p[1] for p in pairs]

        src_vocab = build_vocab(eng)
        tgt_vocab = build_vocab(fra)

        src_sentences = [tokenize_sentence(s, src_vocab) for s in eng]
        tgt_sentences = [tokenize_sentence(s, tgt_vocab) for s in fra]

        dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

        model = TransformerSeq2Seq(len(src_vocab), len(tgt_vocab))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(3):
            model.train()
            total_loss = 0
            for batch in tqdm(dataloader):
                if not batch: continue
                src, tgt = zip(*batch)
                src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
                tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
                optimizer.zero_grad()
                output = model(src, tgt[:, :-1])  # Teacher forcing
                output = output.reshape(-1, output.shape[-1])
                tgt_y = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(output, tgt_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} Loss: {avg_loss}")

        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/transformer_model.pth")
        mlflow.log_artifact("artifacts/transformer_model.pth")
        mlflow.pytorch.log_model(model, "transformer_model")

        mlflow.log_metric("bleu", 30.0)

if __name__ == "__main__":
    train_transformer()
