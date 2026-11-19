# Experimental Configuration

## 1. Dataset and Features

The Tennessee Eastman Process (TEP) dataset was used for model development and evaluation. The models used 52 process variables comprising 41 measured variables and 11 manipulated variables.

Only normal-operation data was used for model training. Metadata columns (`faultNumber`, `simulationRun`, and `sample`) were excluded from model inputs.

Data was divided by simulation runs to prevent observations from the same run appearing across training, validation, and calibration subsets.

The split consisted of:
- Training: 350 runs
- Validation: 75 runs
- Calibration: 75 runs

Random seed: 42.

## 2. Isolation Forest

- Number of trees: 200
- Contamination: auto
- Maximum samples: auto
- Random state: 42
- Training observations: 100,000
- Validation observations: 20,000
- Calibration observations: 20,000
- Anomaly threshold: 0.507832
- Calibration false alarm rate: 1%

## 3. One-Class Support Vector Machine

- Kernel: Radial Basis Function (RBF)
- Nu: 0.01
- Gamma: scale
- Cache size: 256 MB
- Tolerance: 0.001
- Maximum iterations: 10,000
- Training observations: 5,000
- Validation observations: 10,000
- Calibration observations: 10,000
- Anomaly threshold: 0.319226
- Calibration false alarm rate: 1%

## 4. Autoencoder

- Architecture: 52–32–16–8–16–32–52
- Activation: ReLU
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 512
- Maximum epochs: 25
- Early stopping patience: 5
- Training observations: 100,000
- Validation observations: 20,000
- Calibration observations: 20,000
- Anomaly threshold: 0.894699
- Calibration false alarm rate: 1%

The models were trained using CPU computation.

## 5. Threshold Selection

For each algorithm, anomaly scores were calculated on normal calibration observations. The 99th percentile of these scores was selected as the anomaly threshold, targeting a calibration false alarm rate of approximately 1%.

## 6. Evaluation

All three models were evaluated using the same held-out normal and faulty testing datasets. Evaluation included precision, recall, F1-score, specificity, false alarm rate, accuracy, and sampled ROC-AUC.

The evaluation covered 480,000 normal test observations and 9,600,000 faulty-test observations.