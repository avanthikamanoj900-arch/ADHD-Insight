import numpy as np


def normalize_attention_features(features):
    """
    Normalize attention-related time-series features.

    Expected input shape:
        [time_steps, features]

    Returns:
        Normalized NumPy array.
    """

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(
            "Attention features must have shape [time_steps, features]"
        )

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    normalized = (features - mean) / (std + 1e-8)

    return normalized


def create_attention_sequence(
    gaze_direction,
    head_movement,
    eye_movement,
    fixation_duration,
    reaction_time,
    accuracy,
    missed_targets,
    distraction_events
):
    """
    Combine attention measurements into a time-series matrix.

    Each parameter should contain measurements collected
    over multiple time steps.
    """

    features = np.column_stack([
        gaze_direction,
        head_movement,
        eye_movement,
        fixation_duration,
        reaction_time,
        accuracy,
        missed_targets,
        distraction_events
    ])

    return normalize_attention_features(features)
