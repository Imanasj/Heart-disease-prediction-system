"""Load the Kaggle CSV, clean it, store it in SQLite, and read it back with SQL."""
import sqlite3
import pandas as pd
from .config import CSV_PATH, DB_PATH, FEATURES, TARGET

RENAMES = {"thalach": "thalch"}  # older versions of the dataset use 'thalach'


def load_and_clean(csv_path=CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path).rename(columns=RENAMES)
    df.columns = [c.strip().lower() for c in df.columns]

    # 'num' is 0-4 severity -> binary. Some dataset versions already have 'target'.
    if "num" in df.columns:
        df[TARGET] = (df["num"] > 0).astype(int)
    elif TARGET not in df.columns:
        raise ValueError("Expected a 'num' or 'target' column.")

    # Normalise text/booleans so training and the app agree on category names
    for c in ["sex", "cp", "restecg", "slope", "thal", "fbs", "exang"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.lower()
    for c in ["fbs", "exang"]:
        df[c] = df[c].replace({"1": "true", "0": "false", "1.0": "true", "0.0": "false"})

    # Physiologically impossible zeros are really missing in this dataset
    for c in ["chol", "trestbps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == 0, c] = float("nan")

    keep = [c for c in ["id", "dataset", "num"] if c in df.columns] + FEATURES + [TARGET]
    df = df[keep].drop_duplicates(subset=[c for c in keep if c != "id"]).reset_index(drop=True)
    return df


def build_database(df: pd.DataFrame, db_path=DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        df.astype(object).where(df.notna(), None).to_sql("patients", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_target ON patients(target)")
        # Prediction log written by the Streamlit app
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            ts TEXT DEFAULT CURRENT_TIMESTAMP, age REAL, sex TEXT, cp TEXT,
            trestbps REAL, chol REAL, risk REAL, label INTEGER)""")


def query(sql: str, params=(), db_path=DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=params)


def load_training_frame() -> pd.DataFrame:
    """Training data is pulled with SQL."""
    df = query(f"""
        SELECT {", ".join(FEATURES)}, {TARGET}
        FROM patients
        WHERE age IS NOT NULL AND sex IS NOT NULL
    """)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


from .config import NUMERIC as NUMERIC_COLS  # noqa: E402
