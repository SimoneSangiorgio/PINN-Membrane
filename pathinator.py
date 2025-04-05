from pathlib import Path

# Paths
base_directory = Path(__file__).resolve().parent

model_path = base_directory / "model_SimpleSpatioTemporalFFN_10000.pt"

model_test_path = base_directory / "model_test.pt"

csv_simulation_path = base_directory / "SSTFFN_membrane_simulation.csv"

csv_simulation_test_path = base_directory / "membrane_test_simulation.csv"

csv_real_path = base_directory / "membrane.csv"