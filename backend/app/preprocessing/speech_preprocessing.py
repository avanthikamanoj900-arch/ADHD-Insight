import librosa
import numpy as np


def extract_mel_spectrogram(
    audio_path,
    sample_rate=16000,
    n_mels=128,
    duration=5
):
    """
    Convert an audio file into a Mel-spectrogram.

    Parameters:
        audio_path: Path to the audio file
        sample_rate: Target sampling rate
        n_mels: Number of Mel frequency bands
        duration: Maximum audio duration in seconds

    Returns:
        Normalized Mel-spectrogram as a NumPy array
    """

    # Load audio
    audio, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        duration=duration
    )

    # Make sure every input has the same duration
    target_length = sample_rate * duration

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    else:

        audio = audio[:target_length]

    # Extract Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256
    )

    # Convert power to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize
    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    return mel_db.astype(np.float32)
