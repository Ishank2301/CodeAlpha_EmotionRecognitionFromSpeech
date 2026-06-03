import numpy as np
import librosa
import io
import tempfile
import os
from src.utils.config import Config


def load_audio(path, sr=Config.SAMPLE_RATE, duration=Config.DURATION):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def load_audio_bytes(audio_bytes, sr=Config.SAMPLE_RATE, duration=Config.DURATION):
    """Load audio from bytes by writing to a temporary file.

    This approach is more reliable on Windows than using BytesIO directly,
    as soundfile's libsndfile backend has better support for file paths.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(audio_bytes)

    try:
        y, _ = librosa.load(tmp_path, sr=sr, duration=duration)
        target_len = sr * duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def augment_audio(y, sr=Config.SAMPLE_RATE):
    """Light augmentation: original + noise only (2x total) to save memory."""
    augmented = [y]
    noise = y + 0.005 * np.random.randn(len(y))
    augmented.append(noise)
    return augmented


def extract_features(y, sr=Config.SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=Config.N_MFCC,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=Config.N_MELS, hop_length=Config.HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=Config.HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=Config.HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=Config.HOP_LENGTH)

    T = min(mfcc.shape[1], mel_db.shape[1], chroma.shape[1], zcr.shape[1], rms.shape[1])
    features = np.vstack(
        [
            mfcc[:, :T],
            delta[:, :T],
            delta2[:, :T],
            mel_db[:, :T],
            chroma[:, :T],
            zcr[:, :T],
            rms[:, :T],
        ]
    )

    return features.T


def preprocess_for_model(y, scaler, max_len):
    feat = extract_features(y)
    n_feat = feat.shape[1]

    if feat.shape[0] < max_len:
        pad = np.zeros((max_len - feat.shape[0], n_feat))
        feat = np.vstack([feat, pad])
    else:
        feat = feat[:max_len]

    feat_scaled = scaler.transform(feat)
    return feat_scaled[np.newaxis, :, :]
