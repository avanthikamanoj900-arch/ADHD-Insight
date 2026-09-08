import torch
import torch.nn as nn


class AttentionModel(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden_size=64,
        num_classes=4
    ):

        super(AttentionModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(hidden_size * 2, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )


    def forward(self, x):

        # Expected input:
        # [batch, time, features]

        x, _ = self.lstm(x)

        # Self-attention

        attended, _ = self.attention(
            x,
            x,
            x
        )

        # Average across time

        x = attended.mean(dim=1)

        output = self.classifier(x)

        return output
