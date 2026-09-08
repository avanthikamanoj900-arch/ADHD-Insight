import os
import numpy as np


# Number of synthetic participants
NUM_SAMPLES = 200


def generate_speech_data():
    """
    Generate synthetic speech features.
    Shape:
        samples × features × time
    """

    data = np.random.normal(
        0,
        1,
        (NUM_SAMPLES, 128, 100)
    )

    labels = np.random.randint(
        0,
        4,
        NUM_SAMPLES
    )

    return data, labels


def generate_attention_data():
    """
    Generate synthetic attention sequences.

    Shape:
        samples × time × features
    """

    data = np.random.normal(
        0,
        1,
        (NUM_SAMPLES, 20, 8)
    )

    labels = np.random.randint(
        0,
        4,
        NUM_SAMPLES
    )

    return data, labels


def generate_interaction_data():
    """
    Generate synthetic interaction sequences.

    Shape:
        samples × time × features
    """

    data = np.random.normal(
        0,
        1,
        (NUM_SAMPLES, 20, 6)
    )

    labels = np.random.randint(
        0,
        4,
        NUM_SAMPLES
    )

    return data, labels


def generate_learning_data():
    """
    Generate synthetic learning sequences.

    Shape:
        samples × time × features
    """

    data = np.random.normal(
        0,
        1,
        (NUM_SAMPLES, 12, 5)
    )

    labels = np.random.randint(
        0,
        4,
        NUM_SAMPLES
    )

    return data, labels


def save_data():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    speech, speech_labels = generate_speech_data()

    attention, attention_labels = generate_attention_data()

    interaction, interaction_labels = generate_interaction_data()

    learning, learning_labels = generate_learning_data()


    np.save(
        "data/processed/speech.npy",
        speech
    )

    np.save(
        "data/processed/speech_labels.npy",
        speech_labels
    )


    np.save(
        "data/processed/attention.npy",
        attention
    )

    np.save(
        "data/processed/attention_labels.npy",
        attention_labels
    )


    np.save(
        "data/processed/interaction.npy",
        interaction
    )

    np.save(
        "data/processed/interaction_labels.npy",
        interaction_labels
    )


    np.save(
        "data/processed/learning.npy",
        learning
    )

    np.save(
        "data/processed/learning_labels.npy",
        learning_labels
    )


    print("Synthetic datasets generated successfully.")


if __name__ == "__main__":
    save_data()
