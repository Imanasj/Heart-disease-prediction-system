from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "heart_disease_uci.csv"
DB_PATH = ROOT / "heart.db"
MODEL_PATH = ROOT / "heart_xgb.joblib"
REPORTS = ROOT / "reports"
METRICS_PATH = REPORTS / "metrics.json"

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"          # 1 = heart disease present (num > 0)
RANDOM_STATE = 42
