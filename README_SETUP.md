# 🎙️ Speech Emotion Recognition - Complete Project

CNN + BiLSTM + Attention-based emotion recognition from speech using RAVDESS dataset.

## Project Structure

```
CodeAlpha_EmotionRecognitionFromSpeech/
├── app.py                          # Streamlit demo app
├── train.py                        # Training script
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_extractor.py   # Audio feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_loader.py        # Model loading & inference
│   └── utils/
│       ├── __init__.py
│       └── config.py              # Configuration & paths
├── notebooks/
│   └── train_vscode.ipynb         # Training notebook (VSCode)
├── RAVDESS/
│   ├── audio_speech_actors_01-24/ # Dataset (original)
│   └── processed/                 # Processed features & models
└── models/                        # Trained model files
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place RAVDESS audio files in `RAVDESS/audio_speech_actors_01-24/`. Dataset structure:
```
RAVDESS/audio_speech_actors_01-24/
├── Actor_01/
├── Actor_02/
...
└── Actor_24/
```

Download RAVDESS dataset from: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

### 3. Train Model

**Option A: Run training script**
```bash
python train.py
```

**Option B: Run notebook in VSCode**
- Open `notebooks/train_vscode.ipynb`
- Run cells sequentially
- Model will be saved to `models/` and artifacts to `RAVDESS/processed/`

### 4. Run Streamlit App

```bash
streamlit run app.py
```

Upload audio file (WAV, MP3, OGG, FLAC, M4A) to see emotion predictions with confidence scores and visualizations.

## Features Extracted

- **MFCC** (40 coefficients) - spectral characteristics
- **Δ-MFCC** - first order derivatives
- **ΔΔ-MFCC** - second order derivatives  
- **Mel Spectrogram** (128 bins) - perceptual frequency scale
- **Chroma** (12 features) - pitch content
- **ZCR** - zero crossing rate
- **RMS Energy** - amplitude

**Total:** 262-dimensional feature vector

## Model Architecture

**CNN + BiLSTM + Attention**

- 3x Conv1D layers (64→128→256 filters)
- 2x Bidirectional LSTM (128→64 units)
- Attention layer (Bahdanau-style)
- Dense classifier head (256→128 neurons)
- 8 emotion classes

## Emotions Recognized

1. Neutral (😐)
2. Calm (😌)
3. Happy (😄)
4. Sad (😢)
5. Angry (😠)
6. Fearful (😨)
7. Disgust (🤢)
8. Surprised (😲)

## Output Files

After training, check `RAVDESS/processed/`:
- `ser_model.h5` - trained model
- `scaler.pkl` - feature scaler
- `label_encoder.pkl` - emotion encoder
- `config_meta.npy` - model metadata
- `training_log.csv` - training history
- `confusion_matrix.png` - evaluation visualization
- `training_curves.png` - accuracy/loss plots
- `sample_visualization.png` - feature examples

## Usage Example (Python)

```python
from src.models.model_loader import load_artifacts, predict_emotion
from src.features.feature_extractor import load_audio_bytes, preprocess_for_model

artifacts = load_artifacts()
audio_bytes = open('audio.wav', 'rb').read()
y = load_audio_bytes(audio_bytes)
X = preprocess_for_model(y, artifacts['scaler'], artifacts['max_len'])
result = predict_emotion(X, artifacts)

print(result['emotion'], result['confidence'])
```

## Performance

- Test Accuracy: ~85% on RAVDESS
- Training Time: ~2-3 hours (GPU)
- Model Size: ~15MB

## Configuration

Edit `src/utils/config.py`:
- `SAMPLE_RATE = 22050` - audio sampling rate
- `DURATION = 3` - max audio length (seconds)
- `BATCH_SIZE = 32` - training batch size
- `EPOCHS = 80` - max training epochs
- `DROPOUT = 0.4` - regularization
- `LSTM_UNITS = 128` - LSTM hidden size

## Notes

- Dataset augmentation: noise, pitch shift, time stretch
- Feature scaling applied before model input
- Data split: 70% train, 18% val, 12% test
- Early stopping & learning rate reduction enabled
- Best model checkpoint saved automatically

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- librosa 0.10+
- scikit-learn 1.5+
- streamlit 1.45+
- numpy, pandas, matplotlib, seaborn
