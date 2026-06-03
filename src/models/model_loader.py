import os
import pickle
import numpy as np
import tensorflow as tf
from src.utils.config import Config


def load_artifacts():
    scaler_path = os.path.join(Config.PROCESSED_DIR, "scaler.pkl")
    encoder_path = os.path.join(Config.PROCESSED_DIR, "label_encoder.pkl")
    meta_path = os.path.join(Config.PROCESSED_DIR, "config_meta.npy")
    model_path = os.path.join(Config.MODELS_DIR, "ser_model.h5")

    if not all(
        os.path.exists(p) for p in [scaler_path, encoder_path, meta_path, model_path]
    ):
        raise FileNotFoundError("Model artifacts not found. Run train.py first.")

    scaler = pickle.load(open(scaler_path, "rb"))
    le = pickle.load(open(encoder_path, "rb"))
    meta = np.load(meta_path, allow_pickle=True).item()
    model = tf.keras.models.load_model(
        model_path, custom_objects={"AttentionLayer": AttentionLayer}
    )

    return {
        "scaler": scaler,
        "label_encoder": le,
        "model": model,
        "max_len": meta["max_len"],
        "n_feat": meta["n_feat"],
        "n_classes": meta["n_classes"],
    }


def predict_emotion(X, artifacts):
    model = artifacts["model"]
    le = artifacts["label_encoder"]

    y_pred = model.predict(X, verbose=0)
    y_pred_class = np.argmax(y_pred[0])

    emotion = le.classes_[y_pred_class]
    confidence = float(y_pred[0][y_pred_class])

    all_probs = {le.classes_[i]: float(y_pred[0][i]) for i in range(len(le.classes_))}

    return {"emotion": emotion, "confidence": confidence, "all_probs": all_probs}


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, lstm_output):
        score = self.V(tf.nn.tanh(self.W(lstm_output)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * lstm_output, axis=1)
        return context, weights

    def get_config(self):
        return super().get_config()
