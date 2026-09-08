import numpy as np


def normalize_learning_features(features):
    """
    Normalize learning-related time-series features.

    Expected input shape:
        [time_steps, features]

    Returns:
        Normalized NumPy array.
    """

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(
            "Learning features must have shape [time_steps, features]"
        )

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    normalized = (features - mean) / (std + 1e-8)

    return normalized


def create_learning_sequence(
    accuracy,
    response_time,
    improvement,
    retention,
    error_rate
):
    """
    Combine learning measurements into a time-series matrix.

    Each parameter should contain measurements collected
    across multiple learning trials.
    """

    features = np.column_stack([
        accuracy,
        response_time,
        improvement,
        retention,
        error_rate
    ])

    return normalize_learning_features(features)
