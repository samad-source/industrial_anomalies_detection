
from app.services.model_loader import load_models
from app.services.simulation_engine import SimulationEngine
from app.services.simulation_monitor import monitor_simulation_batch


print("=" * 60)
print("SIMULATION AND ANOMALY DETECTION INTEGRATION TEST")
print("=" * 60)

# Load trained models.
models = load_models()

# Create simulation engine.
engine = SimulationEngine(batch_size=1000)

# Load normal operation data.
engine.load_normal_operation()

# Start simulation.
engine.start()

# Read one batch.
batch = engine.next_batch()

# Monitor the batch using all three models.
monitoring = monitor_simulation_batch(models, batch)

results = monitoring["results"]

print("\nMONITORING RESULTS")
print("-" * 60)

print("Simulation source:", monitoring["source"])
print("Observations monitored:", len(results))

for model in [
    "isolation_forest",
    "one_class_svm",
    "autoencoder"
]:
    column = f"{model}_prediction"
    count = int(results[column].sum())

    print(f"{model}: {count} anomalies detected")

print(
    "\nEnsemble alerts:",
    int(results["ensemble_alert"].sum())
)

print(
    "Simulation progress:",
    f'{monitoring["progress"]:.2%}'
)

print("\nINTEGRATION TEST COMPLETED SUCCESSFULLY.")
