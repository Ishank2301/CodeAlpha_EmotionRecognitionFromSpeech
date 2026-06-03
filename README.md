# 🎙️ Speech Emotion Recognition using CNN-BiLSTM

A Deep Learning based Speech Emotion Recognition (SER) system that classifies human emotions from speech audio using acoustic feature extraction and a CNN-BiLSTM neural network architecture.

---

## Project Overview

This project recognizes emotions from speech recordings by extracting:

- MFCC Features
- Chroma Features
- Mel Spectrogram Features

The extracted features are processed using a hybrid:

**CNN + Bidirectional LSTM**

architecture to capture both local acoustic patterns and temporal dependencies in speech signals.

The model is trained on the RAVDESS emotional speech dataset and deployed through an interactive Streamlit web application.

---

## Features

- Speech Emotion Classification
- Audio Feature Extraction using Librosa
- CNN-BiLSTM Deep Learning Model
- Confusion Matrix Generation
- Classification Report
- Training Accuracy & Loss Curves
- Interactive Streamlit UI
- Real-time Emotion Prediction

---

## Supported Emotions

| Emotion | Label |
|----------|----------|
| Neutral | 01 |
| Calm | 02 |
| Happy | 03 |
| Sad | 04 |
| Angry | 05 |
| Fearful | 06 |
| Disgust | 07 |
| Surprised | 08 |

---

## Dataset

RAVDESS

**Ryerson Audio-Visual Database of Emotional Speech and Song**

Dataset contains:

- 24 Professional Actors
- 8 Emotions
- High Quality Speech Recordings

---

## Tech Stack

### Machine Learning

- TensorFlow / Keras
- CNN
- Bidirectional LSTM

### Audio Processing

- Librosa
- NumPy

### Evaluation

- Scikit-Learn
- Matplotlib
- Seaborn

### Deployment

- Streamlit

---

## Project Structure

```text
Speech-Emotion-Recognition/

├── notebooks/
│   └── SER_Training.ipynb

├── src/
│   ├── config.py
│   ├── feature_extraction.py
│   ├── augment.py
│   ├── model.py
│   ├── evaluate.py
│   └── predict.py

├── models/
│   ├── ser_model.keras
│   └── label_encoder.pkl

├── reports/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── classification_report.txt

├── app.py

├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git

cd Speech-Emotion-Recognition
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Run your training notebook or training script.

Model artifacts will be saved in:

```text
models/
```

Generated reports will be saved in:

```text
reports/
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Model Architecture

```text
Audio
  │
  ▼

Feature Extraction
(MFCC + Chroma + Mel)

  │
  ▼

CNN Layers

  │
  ▼

BiLSTM Layer

  │
  ▼

Dense Layers

  │
  ▼

Softmax

  │
  ▼

Emotion Prediction
```

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Future Improvements

- Real-Time Microphone Input
- Transformer-Based SER Models
- Hugging Face Deployment
- Docker Support
- Model Quantization
- Multi-Language Emotion Detection

---

## Author

Ishank Mishra

B.Tech Computer Science Engineering

Machine Learning | Deep Learning | AI Development