from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Resolve the project root from this script's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Fetch Heart Disease dataset from UCI
dataset = fetch_ucirepo(id=45)

# Keep original target values for traceability
df = dataset.data.original.copy()

# Save raw dataset without modifying the target
output_path = RAW_DIR / "heart_disease_raw.csv"
df.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print("Saved to:", output_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())