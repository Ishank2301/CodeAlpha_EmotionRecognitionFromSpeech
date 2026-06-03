# Quick Reference

## One-Command Setup

```bash
cd d:\Ai\ ml\code\ aplpha\CodeAlpha_EmotionRecognitionFromSpeech
pip install -r requirements.txt
python check_setup.py
```

## Training

```bash
python train.py
```

**Duration:** 2-3 hours (GPU) | 6-8 hours (CPU)

**Output:** `RAVDESS/processed/` directory with all artifacts

## Demo

```bash
streamlit run app.py
```

**URL:** http://localhost:8501

## Project Structure

```
CodeAlpha_EmotionRecognitionFromSpeech/
├── app.py                          # Streamlit app
├── train.py                        # Training entry point
├── check_setup.py                  # Setup verification
├── requirements.txt                # Dependencies
├── README_SETUP.md                 # Setup guide
├── USAGE.md                        # Detailed usage
├── QUICK_REF.md                    # This file
├── src/                            # Source code
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_extractor.py    # Audio features
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_loader.py         # Model inference
│   └── utils/
│       ├── __init__.py
│       └── config.py               # Configuration
├── notebooks/
│   ├── train_ser_model.ipynb       # Original (Colab)
│   └── train_vscode.ipynb          # VSCode notebook
├── RAVDESS/
│   ├── audio_speech_actors_01-24/  # Dataset
│   └── processed/                  # Features & models
├── models/                         # Trained models
├── .streamlit/
│   └── config.toml                 # Streamlit config
└── myvenv/                         # Virtual environment
```

## Configuration

All settings in `src/utils/config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| SAMPLE_RATE | 22050 | Hz |
| DURATION | 3 | seconds |
| N_MFCC | 40 | MFCC coefficients |
| N_MELS | 128 | Mel spectrogram bins |
| BATCH_SIZE | 32 | Training batch |
| EPOCHS | 80 | Max epochs |
| LR | 1e-3 | Learning rate |
| DROPOUT | 0.4 | Dropout rate |
| LSTM_UNITS | 128 | LSTM units |

## Key Files

### Training Pipeline
- **Input:** RAVDESS audio files (1440 files)
- **Processing:** Feature extraction + augmentation
- **Output:** `RAVDESS/processed/*`

### Model
- **Type:** CNN + BiLSTM + Attention
- **Input:** (551, 262) feature matrices
- **Output:** 8 emotion probabilities
- **Size:** ~15 MB

### Streamlit App
- **Upload:** Audio files (WAV, MP3, OGG, FLAC, M4A)
- **Display:** Emotion + confidence + all probabilities
- **Visualize:** Waveform + Mel-spectrogram

## Features Extracted

**262 dimensions total:**
- MFCC (40) + Δ-MFCC (40) + ΔΔ-MFCC (40) = 120
- Mel-spectrogram (128)
- Chroma (12)
- ZCR (1) + RMS Energy (1)

## Emotions (8 classes)

```
neutral  😐  | calm     😌
happy    😄  | sad      😢
angry    😠  | fearful  😨
disgust  🤢  | surprised 😲
```

## Output Artifacts

After training, in `RAVDESS/processed/`:

```
✓ ser_model.h5              Main model
✓ scaler.pkl                Feature normalization
✓ label_encoder.pkl         Emotion labels
✓ config_meta.npy           Model metadata
✓ training_log.csv          Training history
✓ confusion_matrix.png      Evaluation metrics
✓ training_curves.png       Accuracy/Loss plots
✓ sample_visualization.png  Feature examples
```

## API Usage

```python
from src.models.model_loader import load_artifacts, predict_emotion
from src.features.feature_extractor import load_audio_bytes, preprocess_for_model

artifacts = load_artifacts()

audio_bytes = open('audio.wav', 'rb').read()
y = load_audio_bytes(audio_bytes)
X = preprocess_for_model(y, artifacts['scaler'], artifacts['max_len'])
result = predict_emotion(X, artifacts)

print(result['emotion'])
print(result['confidence'])
print(result['all_probs'])
```

## Expected Performance

- **Test Accuracy:** ~85%
- **Per-emotion F1:** 0.80-0.92
- **Inference Time:** <100ms per audio

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python train.py` |
| Dataset missing | Download RAVDESS dataset |
| Out of memory | Reduce BATCH_SIZE |
| Low accuracy | Verify training completed |

## Additional Resources

- **Dataset:** https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- **TensorFlow:** https://tensorflow.org
- **Librosa:** https://librosa.org
- **Streamlit:** https://streamlit.io

## Notes

- Processed data stored to avoid recomputation
- Features normalized with StandardScaler
- Data augmentation: noise, pitch shift, time stretch
- Early stopping enabled (patience=12 epochs)
- Learning rate reduction enabled
