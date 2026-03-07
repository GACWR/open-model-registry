"""Sentiment analysis with bidirectional LSTM.

Classifies text sequences into positive/negative sentiment using
a bidirectional LSTM with attention-weighted pooling.
"""

import torch
import torch.nn as nn
import numpy as np


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(self.dropout(context))


def train(ctx):
    hp = ctx.hyperparameters
    epochs = int(hp.get("epochs", 10))
    lr = float(hp.get("lr", 0.001))
    batch_size = int(hp.get("batch_size", 32))
    seq_len = int(hp.get("seq_len", 128))
    vocab_size = int(hp.get("vocab_size", 10000))

    device = ctx.device
    model = SentimentLSTM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ctx.log_metric("progress", 5)

    n_samples = int(hp.get("n_samples", 500))
    X = torch.randint(1, vocab_size, (n_samples, seq_len))
    y = torch.randint(0, 2, (n_samples,))

    ctx.log_metric("progress", 10)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, n_samples, batch_size):
            bx = X[i:i + batch_size].to(device)
            by = y[i:i + batch_size].to(device)

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)

        avg_loss = epoch_loss / max(1, n_samples // batch_size)
        accuracy = correct / max(1, total)
        ctx.log_metric("loss", avg_loss, epoch=epoch)
        ctx.log_metric("accuracy", accuracy, epoch=epoch)
        ctx.log_metric("progress", 10 + int(epoch / epochs * 85))

    ctx.log_metric("progress", 100)


def infer(ctx):
    device = ctx.device
    model = SentimentLSTM().to(device)
    model.eval()

    data = ctx.get_input_data()
    if "sequences" not in data and "features" not in data:
        ctx.set_output({"error": "Provide 'sequences' (token ids) or 'features'"})
        return

    tokens = data.get("sequences", data.get("features"))
    X = torch.tensor(tokens, dtype=torch.long).to(device)
    if X.dim() == 1:
        X = X.unsqueeze(0)

    with torch.no_grad():
        output = model(X)
        probs = torch.softmax(output, dim=1)
        predictions = output.argmax(1).cpu().tolist()
        sentiments = ["negative" if p == 0 else "positive" for p in predictions]

    ctx.set_output({
        "predictions": predictions,
        "sentiments": sentiments,
        "probabilities": probs.cpu().tolist(),
    })
