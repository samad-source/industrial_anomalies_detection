
import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "feature_columns.json"
)


class Autoencoder(nn.Module):
    def __init__(self, input_dim=52):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_models():

    print("Checking required files...")

    required_files = [
        MODELS_DIR / "isolation_forest_run_based.joblib",
        MODELS_DIR / "isolation_forest_run_based_threshold.json",
        MODELS_DIR / "one_class_svm_run_based.joblib",
        MODELS_DIR / "one_class_svm_run_based_threshold.json",
        MODELS_DIR / "autoencoder.pth",
        MODELS_DIR / "autoencoder_threshold.json",
        FEATURE_COLUMNS_PATH
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    print("Loading Isolation Forest...")
    isolation_forest = joblib.load(
        MODELS_DIR / "isolation_forest_run_based.joblib"
    )

    print("Loading One-Class SVM...")
    one_class_svm = joblib.load(
        MODELS_DIR / "one_class_svm_run_based.joblib"
    )

    print("Loading Autoencoder checkpoint...")
    checkpoint = torch.load(
        MODELS_DIR / "autoencoder.pth",
        map_location="cpu",
        weights_only=True
    )

    input_dim = checkpoint["input_dim"]

    autoencoder = Autoencoder(input_dim=input_dim)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    print("Loading anomaly thresholds...")

    thresholds = {
        "isolation_forest": load_json(
            MODELS_DIR / "isolation_forest_run_based_threshold.json"
        )["threshold"],

        "one_class_svm": load_json(
            MODELS_DIR / "one_class_svm_run_based_threshold.json"
        )["threshold"],

        "autoencoder": load_json(
            MODELS_DIR / "autoencoder_threshold.json"
        )["threshold"]
    }

    print("Loading feature configuration...")
    feature_columns = load_json(FEATURE_COLUMNS_PATH)

    print("All three models loaded successfully!")

    return {
        "isolation_forest": isolation_forest,
        "one_class_svm": one_class_svm,
        "autoencoder": autoencoder,
        "thresholds": thresholds,
        "feature_columns": feature_columns
    }
