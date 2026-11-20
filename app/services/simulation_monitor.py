
from app.services.anomaly_detector import detect_anomalies


def monitor_simulation_batch(models, batch):
    """
    Run anomaly detection on one simulation batch.

    Combines process metadata with predictions and anomaly scores
    from all three trained models.
    """

    if batch is None:
        return None

    observations = batch["observations"]

    # Run all three models.
    detection_results = detect_anomalies(
        models,
        observations
    )

    # Copy metadata for the monitoring results.
    results = batch["metadata"].copy()

    # Add predictions and scores.
    for model_name, result in detection_results.items():

        results[f"{model_name}_prediction"] = (
            result["predictions"]
        )

        results[f"{model_name}_score"] = (
            result["scores"]
        )

    # Count how many models identify each observation as anomalous.
    prediction_columns = [
        "isolation_forest_prediction",
        "one_class_svm_prediction",
        "autoencoder_prediction"
    ]

    results["models_flagged"] = results[
        prediction_columns
    ].sum(axis=1)

    results["ensemble_alert"] = (
        results["models_flagged"] >= 2
    ).astype(int)

    return {
        "results": results,
        "detection": detection_results,
        "source": batch["source"],
        "progress": batch["progress"],
        "is_complete": batch["is_complete"]
    }
