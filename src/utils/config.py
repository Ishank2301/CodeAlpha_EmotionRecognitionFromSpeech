import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class Config:
    DATASET_PATH = os.path.join(BASE_DIR, "RAVDESS", "audio_speech_actors_01-24")
    PROCESSED_DIR = os.path.join(BASE_DIR, "RAVDESS", "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    SAMPLE_RATE = 22050
    DURATION = 3
    N_MFCC = 40
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048

    BATCH_SIZE = 32
    EPOCHS = 80
    LR = 1e-3
    DROPOUT = 0.4
    LSTM_UNITS = 128
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_SEED = 42


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)
