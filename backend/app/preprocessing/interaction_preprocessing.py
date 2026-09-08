import numpy as np


def normalize_interaction_features(features):
    """
    Normalize interaction time-series features.

    Expected input shape:
        [time_steps, features]

    Returns:
        Normalized NumPy array.
    """

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(
            "Interaction features must have shape [time_steps, features]"
        )

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    normalized = (features - mean) / (std + 1e-8)

    return normalized


def create_interaction_sequence(
    reaction_time,
    click_accuracy,
    incorrect_clicks,
    missed_responses,
    task_switch_errors,
    response_consistency
):
    """
    Combine interaction measurements into a time-series matrix.
    """

    features = np.column_stack([
        reaction_time,
        click_accuracy,
        incorrect_clicks,
        missed_responses,
        task_switch_errors,
        response_consistency
    ])

    return normalize_interaction_features(features)
