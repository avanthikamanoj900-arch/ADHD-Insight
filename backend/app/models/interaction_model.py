import torch
import torch.nn as nn


class InteractionModel(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_classes=4
    ):

        super(InteractionModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
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

        # Take the final time step

        x = x[:, -1, :]

        output = self.classifier(x)

        return output
