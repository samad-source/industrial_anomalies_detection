# Experimental Configuration

## 1. Dataset and Features

The Tennessee Eastman Process (TEP) benchmark dataset was used for model development and evaluation. The models used 52 process variables comprising 41 measured variables and 11 manipulated variables.

Only normal-operation observations were used for model training. Metadata columns, including `faultNumber`, `simulationRun`, and `sample`, were excluded from model inputs.

Data was divided by simulation runs to prevent observations from the same simulation run appearing across training, validation, and calibration subsets.

The split used three groups:

- Training: 350 simulation runs
- Validation: 75 simulation runs
- Calibration: 75 simulation runs

Random seed: 42.

## 2. Isolation Forest

- Number of estimators: 200
- Contamination: `auto`
- Maximum samples: `auto`
- Random state: 42
- Training observations: 100,000
- Validation observations: 20,000
- Calibration observations: 20,000
- Anomaly threshold: 0.507832
- Calibration percentile: 99th
- Calibration false-alarm rate: 1.00%
- Validation false-alarm rate: 1.235%

The model used parallel CPU processing.

## 3. One-Class Support Vector Machine

- Kernel: Radial Basis Function (RBF)
- Nu: 0.01
- Gamma: `scale`
- Cache size: 256 MB
- Tolerance: 0.001
- Maximum iterations: 10,000
- Training observations: 5,000
- Validation observations: 10,000
- Calibration observations: 10,000
- Anomaly threshold: 0.319226
- Calibration percentile: 99th
- Calibration false-alarm rate: 1.00%
- Validation false-alarm rate: 0.95%

## 4. Autoencoder

- Architecture: 52–32–16–8–16–32–52
- Activation function: ReLU
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 512
- Maximum epochs: 25
- Early stopping patience: 5 epochs
- Training observations: 100,000
- Validation observations: 20,000
- Calibration observations: 20,000
- Anomaly threshold: 0.894699
- Calibration percentile: 99th
- Calibration false-alarm rate: 1.00%

The best-performing model checkpoint, based on validation loss, was restored before saving.

The models were trained using CPU computation.

## 5. Threshold Selection

Anomaly thresholds were selected using the 99th percentile of anomaly scores calculated from normal calibration observations.

For Isolation Forest and One-Class SVM, higher anomaly scores indicate greater abnormality. For the Autoencoder, the anomaly score is the mean squared reconstruction error.

The 99th-percentile calibration threshold targets a false-alarm rate of approximately 1% on normal calibration observations. Actual false-alarm rates may differ on validation and held-out test observations.

## 6. Evaluation

All three models were evaluated using the same held-out normal and faulty-testing datasets.

The evaluation included:

- Precision
- Recall
- F1-score
- Specificity
- False-alarm rate
- Accuracy
- ROC-AUC

The evaluation population consisted of:

- Normal-only test observations: 480,000
- Faulty-testing observations: 9,600,000
- Normal-labelled observations within faulty-testing data: 1,600,000
- Anomalous observations within faulty-testing data: 8,000,000

Therefore, the complete evaluation population contained 10,080,000 observations: 2,080,000 normal and 8,000,000 anomalous.

Confusion-matrix-based classification metrics were calculated using the full evaluation population.

ROC-AUC was estimated using sampled observations because of the size of the dataset:

- Isolation Forest: 130,000 observations
- One-Class SVM: 130,000 observations
- Autoencoder: 482,200 observations

The ROC-AUC sampling procedures differed between models. Consequently, ROC-AUC values should be interpreted with this sampling difference in mind.