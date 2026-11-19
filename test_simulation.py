
from app.services.simulation_engine import SimulationEngine


print("=" * 55)
print("INDUSTRIAL SIMULATION ENGINE TEST")
print("=" * 55)

# Create engine with batches of 1,000 observations.
engine = SimulationEngine(batch_size=1000)

# Load normal operating data.
engine.load_normal_operation()

# Start simulation.
engine.start()

print("\nReading first batch...")

batch = engine.next_batch()

print("Source:", batch["source"])
print("Observations:", batch["observations"].shape)
print("Metadata rows:", len(batch["metadata"]))
print("Progress:", f'{batch["progress"]:.2%}')

# Pause simulation.
engine.pause()

print("\nSimulation paused.")
print("Running:", engine.get_status()["is_running"])

# Resume simulation.
engine.resume()

print("\nSimulation resumed.")

# Read another batch.
batch = engine.next_batch()

print("Second batch:", batch["observations"].shape)
print("Progress:", f'{batch["progress"]:.2%}')

# Reset simulation.
engine.reset()

print("\nSimulation reset.")
print("Current observation:", engine.get_status()["current_observation"])

print("\nSIMULATION ENGINE TEST COMPLETED.")
