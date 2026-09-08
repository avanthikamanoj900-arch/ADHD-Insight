import torch
import torch.nn as nn


class SpeechModel(nn.Module):

    def __init__(self, input_size=128, hidden_size=64, num_classes=4):

        super(SpeechModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
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
        # [batch, features, time]

        x = self.cnn(x)

        # Convert to:
        # [batch, time, features]

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        # Use final timestep

        x = x[:, -1, :]

        output = self.classifier(x)

        return output
