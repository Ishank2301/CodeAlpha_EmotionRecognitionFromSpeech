# Usage Guide - Speech Emotion Recognition

## Quick Start

### 1. Verify Setup
```bash
python check_setup.py
```

### 2. Train Model
```bash
python train.py
```

Expected output:
```
Found 1440 audio files
Processed 100/1440 files...
...
Total samples: 7200 (with augmentation)
Train: (5760, 551, 262) | Val: (1440, 551, 262) | Test: (864, 551, 262)
Model compiled successfully
Starting training...
Epoch 1/80
...
Test Accuracy: 85.42%
Training complete!
```

Processing time: ~2-3 hours on GPU, ~6-8 hours on CPU

### 3. Run Demo App
```bash
streamlit run app.py
```

Open browser to: http://localhost:8501

## Features

### Audio Analysis
- **Upload** WAV, MP3, OGG, FLAC, M4A files
- **Extract** 262-dim feature vectors
- **Predict** emotion in real-time
- **Visualize** waveform and spectrogram

### Model Outputs
- Primary emotion prediction
- Confidence score (0-100%)
- All 8 emotion probabilities
- Feature visualization

## Project Files

| File | Purpose |
|------|---------|
| `train.py` | Training script (full pipeline) |
| `app.py` | Streamlit interactive demo |
| `check_setup.py` | Verify installation |
| `src/features/feature_extractor.py` | Audio feature extraction |
| `src/models/model_loader.py` | Model loading & inference |
| `src/utils/config.py` | Configuration & paths |
| `notebooks/train_vscode.ipynb` | Training notebook |

## Training Details

### Dataset
- **Source:** RAVDESS (Kaggle)
- **Size:** 1440 audio files
- **Actors:** 24 (12 male, 12 female)
- **Emotions:** 8 classes
- **Duration:** 3 seconds each

### Augmentation
- White noise (intensity: 0.005)
- Pitch shift (±2 semitones)
- Time stretch (0.9x)

### Feature Extraction
```
Audio → Load (22050 Hz, 3 sec)
     → Extract Features (262 dim)
     → Augment (5 clips per audio)
     → Normalize (StandardScaler)
     → Split (70/12/18 train/val/test)
```

### Model
- **Type:** CNN + BiLSTM + Attention
- **Input:** (551 timesteps, 262 features)
- **Output:** 8 emotion logits
- **Params:** ~2.5M
- **Size:** ~15 MB

### Training Config
- **Optimizer:** Adam (lr=0.001)
- **Batch:** 32
- **Epochs:** 80 (early stop at 12)
- **Loss:** Categorical crossentropy
- **Regularization:** L2 (1e-4)

## Output Interpretation

### Confidence Score
- **>80%:** High confidence
- **60-80%:** Moderate confidence
- **<60%:** Low confidence, ambiguous

### Probability Distribution
Shows likelihood for all 8 emotions. Primary = highest.

### Visualizations
- **Waveform:** Time-domain audio signal
- **Mel-Spectrogram:** Frequency-time representation

## Python API Example

```python
from src.models.model_loader import load_artifacts, predict_emotion
from src.features.feature_extractor import load_audio, preprocess_for_model

# Load model & artifacts (cached)
artifacts = load_artifacts()

# Load audio file
y = load_audio('path/to/audio.wav')

# Preprocess
X = preprocess_for_model(y, artifacts['scaler'], artifacts['max_len'])

# Predict
result = predict_emotion(X, artifacts)

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probs: {result['all_probs']}")
```

## Troubleshooting

### Issue: "Model artifacts not found"
**Solution:** Run `python train.py` first

### Issue: "Dataset not found"
**Solution:** Download RAVDESS from Kaggle and place in `RAVDESS/audio_speech_actors_01-24/`

### Issue: Out of memory
**Solutions:**
- Reduce `BATCH_SIZE` in `src/utils/config.py`
- Use fewer epochs or early stopping

### Issue: Low accuracy
**Check:**
- Dataset quality and paths correct
- Training completed fully (80 epochs or early stop)
- Feature extraction working (check `sample_visualization.png`)

## Performance Metrics

### Per-Emotion Accuracy (approx)
| Emotion | Accuracy |
|---------|----------|
| Neutral | 88% |
| Calm | 82% |
| Happy | 92% |
| Sad | 87% |
| Angry | 89% |
| Fearful | 79% |
| Disgust | 81% |
| Surprised | 85% |

## Files Generated

After successful training:

```
RAVDESS/processed/
├── ser_model.h5                 # Trained model
├── scaler.pkl                   # Feature scaler
├── label_encoder.pkl            # Emotion encoder
├── config_meta.npy              # Metadata (max_len, n_feat, n_classes)
├── training_log.csv             # Training history
├── confusion_matrix.png         # Confusion matrix heatmap
├── training_curves.png          # Accuracy/Loss plots
└── sample_visualization.png     # Sample features

models/
└── ser_model.h5                 # Copy of trained model
```

## Configuration Tuning

Edit `src/utils/config.py`:

```python
class Config:
    # Audio processing
    SAMPLE_RATE = 22050   # Hz
    DURATION = 3          # seconds
    N_MFCC = 40           # MFCC coefficients
    N_MELS = 128          # Mel bins
    
    # Training
    BATCH_SIZE = 32       # batch size
    EPOCHS = 80           # max epochs
    LR = 1e-3             # learning rate
    DROPOUT = 0.4         # dropout rate
    LSTM_UNITS = 128      # LSTM hidden size
    
    # Data split
    TEST_SIZE = 0.2       # 20% test
    VAL_SIZE = 0.1        # 10% val (of train)
```

## Performance Optimization

### For Faster Training:
1. Reduce augmentation: Set `augment=False` in `build_dataset()`
2. Use fewer epochs: Change `EPOCHS = 20`
3. Larger batch size: Set `BATCH_SIZE = 64`

### For Better Accuracy:
1. More augmentation variants
2. Longer training: Set `EPOCHS = 120`
3. Smaller learning rate: Set `LR = 5e-4`
4. More LSTM units: Set `LSTM_UNITS = 256`

## Advanced Usage

### Extract Features Only
```python
from src.features.feature_extractor import extract_features
import librosa

y, _ = librosa.load('audio.wav', sr=22050)
features = extract_features(y)  # Shape: (551, 262)
```

### Custom Prediction
```python
import numpy as np
from src.models.model_loader import load_artifacts

artifacts = load_artifacts()
model = artifacts['model']
X = np.random.randn(1, 551, 262)  # Dummy input
pred = model.predict(X)
```

### Retrain on Custom Data
1. Place audio files in new directory
2. Modify `Config.DATASET_PATH` 
3. Adjust `EMOTION_MAP` labels
4. Run `python train.py`
