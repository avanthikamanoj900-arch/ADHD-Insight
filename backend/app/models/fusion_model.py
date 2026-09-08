import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):

    def __init__(
        self,
        speech_size=64,
        attention_size=64,
        interaction_size=64,
        learning_size=64,
        fusion_size=128,
        output_size=6
    ):

        super(MultimodalFusionModel, self).__init__()

        # Combine the four modality embeddings
        total_size = (
            speech_size
            + attention_size
            + interaction_size
            + learning_size
        )

        self.fusion = nn.Sequential(

            nn.Linear(total_size, fusion_size),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(fusion_size, 64),

            nn.ReLU()
        )

        # Behavioral profile outputs
        self.profile = nn.Linear(
            64,
            output_size
        )


    def forward(
        self,
        speech_embedding,
        attention_embedding,
        interaction_embedding,
        learning_embedding
    ):

        # Combine all four embeddings

        combined = torch.cat(
            [
                speech_embedding,
                attention_embedding,
                interaction_embedding,
                learning_embedding
            ],
            dim=1
        )

        # Learn relationships between modalities

        fused = self.fusion(combined)

        # Generate behavioral profile

        output = self.profile(fused)

        return output
