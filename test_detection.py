
import numpy as np

from app.services.model_loader import load_models
from app.services.anomaly_detector import detect_anomalies


print("=" * 55)
print("INTEGRATED ANOMALY DETECTION TEST")
print("=" * 55)

# Load trained models
models = load_models()

# Load 1,000 already-standardized normal observations
X = np.load(
    "data/processed/X_test_normal.npy",
    mmap_mode="r"
)[:1000].copy()

print(f"\nObservations loaded: {len(X)}")
print(f"Features per observation: {X.shape[1]}")

# Run all three models
results = detect_anomalies(models, X)

print("\nDETECTION RESULTS")
print("-" * 55)

for model_name, result in results.items():

    anomaly_count = int(result["predictions"].sum())
    mean_score = float(result["scores"].mean())

    print(f"\nModel: {model_name}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Normal predictions: {len(X) - anomaly_count}")
    print(f"Mean anomaly score: {mean_score:.6f}")

print("\n" + "=" * 55)
print("ALL THREE MODELS EXECUTED SUCCESSFULLY.")
print("=" * 55)
