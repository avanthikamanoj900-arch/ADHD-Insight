import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from backend.app.models.speech_model import SpeechModel


# -----------------------------
# Configuration
# -----------------------------

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001


# -----------------------------
# Load synthetic dataset
# -----------------------------

X = np.load(
    "data/processed/speech.npy"
)

y = np.load(
    "data/processed/speech_labels.npy"
)


# Convert NumPy → PyTorch

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)


# -----------------------------
# Create DataLoader
# -----------------------------

dataset = TensorDataset(
    X,
    y
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -----------------------------
# Create model
# -----------------------------

model = SpeechModel(
    input_size=128,
    hidden_size=64,
    num_classes=4
)


# -----------------------------
# Loss + optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

print("Starting Speech Model training...")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for inputs, labels in loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# -----------------------------
# Save trained model
# -----------------------------

os.makedirs(
    "models/speech",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/speech/speech_model.pth"
)


print(
    "Speech model training completed."
)

print(
    "Model saved to "
    "models/speech/speech_model.pth"
)
