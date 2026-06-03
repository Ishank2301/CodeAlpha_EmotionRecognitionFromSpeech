import os
import glob
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.utils import to_categorical

from src.utils.config import Config, EMOTION_MAP
from src.features.feature_extractor import load_audio, augment_audio, extract_features

warnings.filterwarnings("ignore")


def parse_ravdess_label(filepath):
    fname = os.path.basename(filepath)
    parts = fname.replace(".wav", "").split("-")
    return EMOTION_MAP.get(parts[2], None)


def build_dataset(use_augmentation=True):
    audio_files = glob.glob(
        os.path.join(Config.DATASET_PATH, "**", "*.wav"), recursive=True
    )
    print(f"Found {len(audio_files)} audio files")

    X, y_labels = [], []
    for i, fpath in enumerate(audio_files):
        label = parse_ravdess_label(fpath)
        if label is None:
            continue
        try:
            audio = load_audio(fpath)
            clips = augment_audio(audio) if use_augmentation else [audio]
            for clip in clips:
                feat = extract_features(clip)
                X.append(feat)
                y_labels.append(label)
        except Exception as e:
            print(f"Error on {fpath}: {e}")
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(audio_files)} files...")

    print(f"Total samples: {len(X)}")
    return X, y_labels


class AttentionLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, lstm_output):
        score = self.V(tf.nn.tanh(self.W(lstm_output)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * lstm_output, axis=1)
        return context, weights


def build_model(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape, name="audio_features")

    x = layers.Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(Config.DROPOUT)(x)

    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(Config.DROPOUT)(x)

    x = layers.Conv1D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(Config.DROPOUT * 0.75)(x)

    x = layers.Bidirectional(
        layers.LSTM(
            Config.LSTM_UNITS,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_1",
    )(x)
    x = layers.Dropout(Config.DROPOUT)(x)

    x = layers.Bidirectional(
        layers.LSTM(
            Config.LSTM_UNITS // 2,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_2",
    )(x)

    attention = AttentionLayer(units=64, name="attention")
    context, attn_weights = attention(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(
        context
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(Config.DROPOUT)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(Config.DROPOUT * 0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(
        inputs=inputs, outputs=outputs, name="SER_CNN_BiLSTM_Attention"
    )
    return model


def train():
    print("Loading dataset...")
    X_raw, y_raw = build_dataset(use_augmentation=True)

    print("Preprocessing...")
    MAX_LEN = max(x.shape[0] for x in X_raw)
    N_FEAT = X_raw[0].shape[1]
    print(f"Max time steps: {MAX_LEN}, Feature dim: {N_FEAT}")

    def pad_sequence(x, max_len, n_feat):
        if x.shape[0] < max_len:
            pad = np.zeros((max_len - x.shape[0], n_feat))
            return np.vstack([x, pad])
        return x[:max_len]

    X = np.array([pad_sequence(x, MAX_LEN, N_FEAT) for x in X_raw])

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    y_cat = to_categorical(y_enc)
    N_CLASSES = len(le.classes_)
    print(f"Classes: {le.classes_}")

    X_flat = X.reshape(-1, N_FEAT)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X = X_flat_scaled.reshape(X.shape).astype(
        np.float32
    )  # Convert to float32 to save memory

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y_enc,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=Config.VAL_SIZE / (1 - Config.TEST_SIZE),
        random_state=Config.RANDOM_SEED,
    )

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    print("Building model...")
    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
    model = build_model(INPUT_SHAPE, N_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = os.path.join(Config.MODELS_DIR, "ser_model.h5")

    cb_list = [
        callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(os.path.join(Config.PROCESSED_DIR, "training_log.csv")),
    ]

    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    print("Evaluating...")
    model.load_weights(model_path)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(
        classification_report(y_true_classes, y_pred_classes, target_names=le.classes_)
    )

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PROCESSED_DIR, "confusion_matrix.png"), dpi=150)
    print(f"Saved: confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Acc", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PROCESSED_DIR, "training_curves.png"), dpi=150)
    print(f"Saved: training_curves.png")

    pickle.dump(scaler, open(os.path.join(Config.PROCESSED_DIR, "scaler.pkl"), "wb"))
    pickle.dump(le, open(os.path.join(Config.PROCESSED_DIR, "label_encoder.pkl"), "wb"))
    np.save(
        os.path.join(Config.PROCESSED_DIR, "config_meta.npy"),
        {"max_len": MAX_LEN, "n_feat": N_FEAT, "n_classes": N_CLASSES},
    )

    print("\nTraining complete!")
    print(f"Model saved to: {Config.MODELS_DIR}")
    print(f"Artifacts saved to: {Config.PROCESSED_DIR}")


if __name__ == "__main__":
    train()
