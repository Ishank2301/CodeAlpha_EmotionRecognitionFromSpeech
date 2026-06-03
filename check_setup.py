#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config, EMOTION_MAP
from src.features.feature_extractor import load_audio, extract_features
from src.models.model_loader import load_artifacts, predict_emotion
from src.features.feature_extractor import preprocess_for_model


def check_setup():
    print("🔍 Checking project setup...\n")

    checks = {
        "Dataset Path": os.path.exists(Config.DATASET_PATH),
        "Processed Dir": os.path.exists(Config.PROCESSED_DIR),
        "Models Dir": os.path.exists(Config.MODELS_DIR),
        "Feature Extractor": os.path.exists("src/features/feature_extractor.py"),
        "Model Loader": os.path.exists("src/models/model_loader.py"),
        "Config": os.path.exists("src/utils/config.py"),
    }

    for name, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"{status} {name}")

    print(f"\n📊 Emotions: {list(EMOTION_MAP.values())}")

    print("\n🎯 To start:")
    print("  1. python train.py          (train model)")
    print("  2. streamlit run app.py     (run demo)")

    return all(checks.values())


if __name__ == "__main__":
    if check_setup():
        print("\n✅ Setup looks good!")
    else:
        print("\n⚠️  Some checks failed. Ensure dataset is in place.")
        sys.exit(1)
