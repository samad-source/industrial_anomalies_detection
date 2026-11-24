# Anomaly Detection in Industrial Processes Using Machine Learning

**A machine-learning research prototype for detecting abnormal behaviour in industrial process data.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-green)](https://scikit-learn.org/)
[![Research](https://img.shields.io/badge/Project-Academic%20Research-purple)]()

## 1. Project Overview

Industrial processes generate large volumes of operational data through sensors and control systems. Identifying abnormal process behaviour early is important for operational awareness, reliability and process monitoring.

This project investigates the application of three machine-learning approaches to industrial anomaly detection:

- Isolation Forest
- One-Class Support Vector Machine (One-Class SVM)
- Autoencoder

The models are trained using normal-operation observations and evaluated using normal and faulty observations from the Tennessee Eastman Process (TEP) benchmark dataset.

The project also includes an interactive Streamlit application that presents model predictions through a conceptual industrial facility monitoring environment.

The interface uses an oil-and-gas-style facility representation to make the results easier to understand. However, the underlying experimental dataset is the Tennessee Eastman chemical-process benchmark, not actual oil-and-gas facility data.

## 2. Research Objectives

The main objectives are to:

1. Investigate machine-learning techniques for detecting abnormal behaviour in industrial processes.
2. Train anomaly detection models using normal-operation data.
3. Apply simulation-run-based data splitting to reduce data leakage between training, validation and calibration subsets.
4. Establish anomaly detection thresholds using normal calibration observations.
5. Evaluate and compare model performance using classification metrics and ROC-AUC.
6. Investigate model performance across individual fault categories.
7. Develop an interactive monitoring application for presenting anomaly detection results.

## 3. Dataset

### Tennessee Eastman Process

The project uses the Tennessee Eastman Process benchmark dataset.

The dataset represents a simulated chemical manufacturing process and contains normal and faulty operating conditions.

The models use 52 process variables:

- 41 measured variables
- 11 manipulated variables

Metadata columns such as `faultNumber`, `simulationRun` and `sample` are excluded from model inputs.

### Evaluation dataset

The evaluation population contains:

| Category | Observations |
|---|---:|
| Normal-only test observations | 480,000 |
| Faulty-testing observations | 9,600,000 |
| Normal-labelled observations within faulty-testing data | 1,600,000 |
| Anomalous observations within faulty-testing data | 8,000,000 |
| **Total normal observations** | **2,080,000** |
| **Total anomalous observations** | **8,000,000** |
| **Total evaluation observations** | **10,080,000** |

The faulty-testing dataset contains both normal-labelled and anomalous observations. It should therefore not be interpreted as consisting entirely of anomalies.

## 4. Machine Learning Models

### 4.1 Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm that identifies observations that are easier to isolate from the rest of the data.

The run-based experiment uses:

- 200 estimators
- Automatic contamination setting
- Automatic maximum sample setting
- 100,000 training observations
- 99th-percentile calibration threshold

### 4.2 One-Class Support Vector Machine

One-Class SVM learns a boundary around normal observations and identifies observations that fall outside the learned normal region.

The experiment uses:

- Radial Basis Function (RBF) kernel
- Nu: 0.01
- Gamma: scale
- 5,000 training observations
- 99th-percentile calibration threshold

### 4.3 Autoencoder

The Autoencoder is a neural network trained to reconstruct normal process observations.

Anomalous behaviour is identified using reconstruction error. Observations with reconstruction errors at or above the selected threshold are classified as anomalies.

Architecture:

`52 → 32 → 16 → 8 → 16 → 32 → 52`

Configuration:

- Activation: ReLU
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 512
- Maximum epochs: 25
- Early stopping patience: 5
- 100,000 training observations
- 99th-percentile calibration threshold

## 5. Experimental Methodology

### Run-based data separation

The normal-operation data is divided using simulation-run-based splitting.

| Subset | Simulation runs |
|---|---:|
| Training | 350 |
| Validation | 75 |
| Calibration | 75 |

Random seed: 42.

The purpose of this approach is to prevent observations from the same simulation run from appearing in different subsets.

### Threshold calibration

Each model uses the 99th percentile of anomaly scores from normal calibration observations to establish its anomaly threshold.

This targets an approximately 1% calibration false-alarm rate.

The calibration target is not a guarantee of the same false-alarm rate on unseen evaluation data.

### Evaluation metrics

The models are evaluated using:

- Precision
- Recall
- F1-score
- Specificity
- False-alarm rate
- Accuracy
- ROC-AUC

Confusion-matrix-based classification metrics are calculated using the complete evaluation population.

ROC-AUC is estimated using sampled observations because of the large dataset size:

| Model | ROC-AUC sample size |
|---|---:|
| Isolation Forest | 130,000 |
| One-Class SVM | 130,000 |
| Autoencoder | 482,200 |

The sampling procedures differ between models. This should be considered when interpreting ROC-AUC comparisons.

## 6. Experimental Results

### Overall model comparison

The following results were obtained from the run-based Isolation Forest and One-Class SVM experiments and the Autoencoder experiment.

| Metric | Isolation Forest | One-Class SVM | Autoencoder |
|---|---:|---:|---:|
| Precision | 99.49% | 99.72% | 99.69% |
| Recall | 38.05% | 62.20% | 62.18% |
| F1-score | 55.05% | 76.61% | 76.59% |
| Specificity | 99.25% | 99.33% | 99.25% |
| False-alarm rate | 0.75% | 0.67% | 0.75% |
| Accuracy | 50.68% | 69.86% | 69.83% |
| ROC-AUC | 83.01% | 88.32% | 86.52% |

### Initial interpretation

Within this benchmark evaluation:

- One-Class SVM achieved the highest precision, recall, F1-score and ROC-AUC.
- The Autoencoder achieved a very similar F1-score and recall to One-Class SVM.
- Isolation Forest achieved high precision and specificity but lower recall.
- One-Class SVM recorded the lowest false-alarm rate among the three models.

These findings describe performance on the selected benchmark evaluation population. They do not establish general superiority in real industrial facilities.

### Fault-wise analysis

The project also investigates model performance across 20 fault categories.

Fault-wise analysis helps identify differences in model detection performance across individual fault types rather than relying exclusively on overall averages.

The resulting comparisons and visualizations are available in the `reports/` directory.

## 7. Interactive Monitoring Application

The project includes a Streamlit-based application designed to make the experimental results accessible through an interactive monitoring interface.

The application provides:

- A conceptual industrial facility interface.
- Three-dimensional facility visualization.
- Loading and use of the trained anomaly detection models.
- Process simulation and anomaly detection.
- Model prediction and anomaly-score monitoring.
- Alarm event recording.
- Alarm management and acknowledgement functionality.
- Visual presentation of detected abnormal conditions.

The application uses prepared benchmark observations to demonstrate monitoring behaviour.

**Important:** This is not a live industrial monitoring system. It does not receive actual operational sensor feeds, and its facility visualization is not a physically validated digital twin.

## 8. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | Data processing and analysis |
| Scikit-learn | Isolation Forest and One-Class SVM |
| PyTorch | Autoencoder development |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive application |
| Plotly | Interactive visualizations |
| Joblib | Model serialization |
| SQLite | Local alarm event storage |
| Git and GitHub | Version control and source management |

## 9. Project Structure

```text
industrial_anomaly_detection/
│
├── app/
│   ├── main.py
│   ├── components/
│   │   └── facility_3d.py
│   └── services/
│       ├── alarm_database.py
│       ├── alarm_manager.py
│       ├── anomaly_detector.py
│       ├── model_loader.py
│       ├── simulation_engine.py
│       └── simulation_monitor.py
│
├── backend/
├── frontend/
│
├── data/
│   └── processed/
│
├── docs/
│   └── experimental_configuration.md
│
├── models/
│   ├── autoencoder.pth
│   ├── autoencoder_threshold.json
│   ├── isolation_forest_run_based.joblib
│   ├── isolation_forest_run_based_threshold.json
│   ├── one_class_svm_run_based.joblib
│   └── one_class_svm_run_based_threshold.json
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── reports/
│   ├── figures/
│   ├── run_based_model_comparison.csv
│   ├── run_based_faultwise_comparison.csv
│   └── publication results and evaluation reports
│
├── src/
│   ├── prepare_data.py
│   ├── train_isolation_forest.py
│   ├── train_one_class_svm.py
│   ├── train_autoencoder.py
│   ├── evaluate_isolation_forest.py
│   ├── evaluate_one_class_svm.py
│   ├── evaluate_autoencoder.py
│   └── visualization and analysis scripts
│
├── tests/
├── requirements.txt
└── README.md
```

## 10. Installation and Local Execution

### Prerequisites

- Windows, Linux or macOS
- Python 3.11
- Git
- Visual Studio Code (recommended)

### Clone the repository

```bash
git clone https://github.com/samad-source/industrial_anomalies_detection.git

cd industrial_anomaly_detection
```

### Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Run the application

From the project root:

```powershell
streamlit run app/main.py
```

Streamlit will provide a local address for opening the application in a web browser.

The trained models and any required prepared data files must be available in their expected project locations.

## 11. Deployment

The interactive application has been deployed using Streamlit Community Cloud.

The deployed application demonstrates the trained models and benchmark-driven monitoring interface.

For persistent alarm history, a managed database should be used. Local SQLite storage on an ephemeral hosting filesystem should not be assumed to persist through application restarts or redeployments.

## 12. Research Limitations

This project is an academic research prototype.

Important limitations include:

1. **Benchmark dataset:** The experiments use the Tennessee Eastman Process chemical-process benchmark rather than real oil-and-gas operational data.
2. **Conceptual facility representation:** The oil-and-gas-style interface is a visualization concept, not a physically validated digital twin.
3. **Simulation-based monitoring:** The application replays prepared benchmark observations rather than monitoring live industrial sensors.
4. **ROC-AUC sampling:** Different ROC-AUC sampling procedures and sample sizes were used across the models.
5. **Generalization:** Performance on benchmark data does not establish performance on unseen real-world industrial facilities.
6. **Operational deployment:** Real industrial use would require facility-specific data validation, cybersecurity assessment, engineering review and appropriate safety oversight.

The application should not be used to make real-world process-control or safety-critical decisions.

## 13. Future Work

Potential future research and development directions include:

- Evaluation using real industrial process datasets.
- Testing model generalization across different operating conditions.
- Further investigation of difficult-to-detect fault categories.
- Improving anomaly detection performance and reducing missed detections.
- Incorporating additional machine-learning and deep-learning approaches.
- Persistent database integration for alarm history.
- Integration with authorized industrial data sources in a controlled research environment.
- More realistic process visualization and digital-twin development.
- Additional validation before any operational deployment.

## 14. Reproducibility

The experimental configuration, model thresholds, evaluation metrics, training histories and visualizations are maintained within the project repository.

Key documentation:

- `docs/experimental_configuration.md`
- `reports/`
- `models/`
- `src/`

Random seed: 42.

## 15. Repository and Live website

**GitHub:** https://github.com/samad-source/industrial_anomalies_detection
**Live Website:** https://industrial-anomalydetection.streamlit.app/
---

*An academic research project investigating machine-learning approaches to anomaly detection in industrial processes.*