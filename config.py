from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "heart_disease_uci.csv"
DB_PATH = ROOT / "data" / "heart.db"
MODEL_PATH = ROOT / "models" / "heart_xgb.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORTS = ROOT / "reports"

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"          # 1 = heart disease present (num > 0)
RANDOM_STATE = 42
