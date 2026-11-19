
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FAULTY_DIR = PROCESSED_DIR / "faulty_testing"


class SimulationEngine:
    """
    Dataset-driven industrial process simulation engine.

    Reads standardized process observations in batches.
    Supports normal operation and faulty scenario replay.
    """

    def __init__(self, batch_size=1000):

        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")

        self.batch_size = batch_size

        self.features = None
        self.metadata = None

        self.source = None
        self.cursor = 0
        self.total_observations = 0

        self.is_running = False

    def load_normal_operation(self):

        feature_path = PROCESSED_DIR / "X_test_normal.npy"
        metadata_path = PROCESSED_DIR / "test_metadata.csv"

        self._load_dataset(
            feature_path,
            metadata_path,
            source="normal"
        )

    def load_fault_scenario(self, chunk_number):

        if not 1 <= chunk_number <= 24:
            raise ValueError(
                "Fault chunk number must be between 1 and 24."
            )

        chunk_name = f"faulty_testing_{chunk_number:02d}"

        feature_path = FAULTY_DIR / f"{chunk_name}_features.npy"
        metadata_path = FAULTY_DIR / f"{chunk_name}_metadata.csv"

        self._load_dataset(
            feature_path,
            metadata_path,
            source=f"fault_chunk_{chunk_number:02d}"
        )

    def _load_dataset(self, feature_path, metadata_path, source):

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        # Memory-map the NumPy array instead of loading it all into RAM.
        features = np.load(feature_path, mmap_mode="r")

        if features.ndim != 2 or features.shape[1] != 52:
            raise ValueError(
                "Expected a dataset containing 52 process features."
            )

        # Metadata for the selected dataset.
        metadata = pd.read_csv(metadata_path)

        if len(features) != len(metadata):
            raise ValueError(
                "Feature and metadata row counts do not match."
            )

        self.features = features
        self.metadata = metadata

        self.source = source
        self.cursor = 0
        self.total_observations = len(features)

        self.is_running = False

        print(f"Loaded simulation source: {source}")
        print(f"Observations available: {self.total_observations:,}")

    def start(self):

        if self.features is None:
            raise RuntimeError(
                "Load a simulation dataset before starting."
            )

        self.cursor = 0
        self.is_running = True

    def pause(self):

        self.is_running = False

    def resume(self):

        if self.features is None:
            raise RuntimeError("No simulation dataset is loaded.")

        if self.cursor >= self.total_observations:
            raise RuntimeError(
                "Simulation has finished. Reset it before resuming."
            )

        self.is_running = True

    def reset(self):

        self.cursor = 0
        self.is_running = False

    def next_batch(self):

        if not self.is_running:
            return None

        if self.cursor >= self.total_observations:
            self.is_running = False
            return None

        start = self.cursor

        end = min(
            start + self.batch_size,
            self.total_observations
        )

        observations = np.asarray(
            self.features[start:end],
            dtype=np.float32
        )

        batch_metadata = self.metadata.iloc[start:end].copy()

        self.cursor = end

        if self.cursor >= self.total_observations:
            self.is_running = False

        return {
            "observations": observations,
            "metadata": batch_metadata,
            "source": self.source,
            "batch_start": start,
            "batch_end": end,
            "total_observations": self.total_observations,
            "progress": self.cursor / self.total_observations,
            "is_complete": self.cursor >= self.total_observations
        }

    def get_status(self):

        return {
            "source": self.source,
            "is_running": self.is_running,
            "current_observation": self.cursor,
            "total_observations": self.total_observations,
            "progress": (
                self.cursor / self.total_observations
                if self.total_observations else 0.0
            )
        }
