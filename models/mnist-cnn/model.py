"""MNIST digit classification with a simple CNN.

A beginner-friendly PyTorch model for handwritten digit recognition
using two convolutional layers followed by fully connected layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(ctx):
    hp = ctx.hyperparameters
    epochs = int(hp.get("epochs", 5))
    lr = float(hp.get("lr", 0.001))
    batch_size = int(hp.get("batch_size", 64))

    device = ctx.device
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ctx.log_metric("progress", 5)
    ctx.log_metric("parameters", float(sum(p.numel() for p in model.parameters())))

    # Synthetic MNIST-like data
    n_samples = int(hp.get("n_samples", 1000))
    X = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 10, (n_samples,))

    ctx.log_metric("progress", 10)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, n_samples, batch_size):
            batch_x = X[i:i + batch_size].to(device)
            batch_y = y[i:i + batch_size].to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = epoch_loss / max(1, n_samples // batch_size)
        accuracy = correct / max(1, total)
        ctx.log_metric("loss", avg_loss, epoch=epoch)
        ctx.log_metric("accuracy", accuracy, epoch=epoch)
        ctx.log_metric("progress", 10 + int(epoch / epochs * 85))

    ctx.log_metric("progress", 100)


def infer(ctx):
    device = ctx.device
    model = MNISTNet().to(device)
    model.eval()

    data = ctx.get_input_data()
    if "features" not in data:
        ctx.set_output({"error": "No 'features' key in input_data"})
        return

    X = torch.tensor(data["features"], dtype=torch.float32)
    if X.dim() == 2:
        X = X.view(-1, 1, 28, 28)
    elif X.dim() == 3:
        X = X.unsqueeze(1)

    X = X.to(device)
    with torch.no_grad():
        output = model(X)
        probs = F.softmax(output, dim=1)
        predictions = output.argmax(dim=1).cpu().tolist()
        confidences = probs.max(dim=1).values.cpu().tolist()

    ctx.set_output({
        "predictions": predictions,
        "confidences": confidences,
    })
