
import numpy as np
import torch


def detect_anomalies(models, observations):
    """
    Run all three trained models on the same observations.

    Parameters
    ----------
    models : dict
        Models returned by load_models().

    observations : numpy.ndarray
        Scaled observations with 52 process features.

    Returns
    -------
    dict
        Anomaly predictions and anomaly scores.
    """

    observations = np.array(
    observations,
    dtype=np.float32,
    copy=True
)
    if observations.ndim == 1:
        observations = observations.reshape(1, -1)

    if observations.ndim != 2 or observations.shape[1] != 52:
        raise ValueError(
            "Expected observations with exactly 52 process features."
        )

    thresholds = models["thresholds"]

    # Isolation Forest
    isolation_scores = -models["isolation_forest"].score_samples(
        observations
    )

    isolation_predictions = (
        isolation_scores > thresholds["isolation_forest"]
    ).astype(int)

    # One-Class SVM
    svm_scores = -models["one_class_svm"].score_samples(
        observations
    )

    svm_predictions = (
        svm_scores > thresholds["one_class_svm"]
    ).astype(int)

    # Autoencoder reconstruction errors
    autoencoder = models["autoencoder"]

    tensor_data = torch.from_numpy(observations)

    with torch.no_grad():
        reconstructed = autoencoder(tensor_data)

        reconstruction_errors = torch.mean(
            (tensor_data - reconstructed) ** 2,
            dim=1
        ).numpy()

    autoencoder_predictions = (
        reconstruction_errors > thresholds["autoencoder"]
    ).astype(int)

    return {
        "isolation_forest": {
            "scores": isolation_scores,
            "predictions": isolation_predictions
        },

        "one_class_svm": {
            "scores": svm_scores,
            "predictions": svm_predictions
        },

        "autoencoder": {
            "scores": reconstruction_errors,
            "predictions": autoencoder_predictions
        }
    }
