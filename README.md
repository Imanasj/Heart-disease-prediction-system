# Heart Disease Prediction System
XGBoost + pandas + SQLite (SQL) + Streamlit on the UCI Heart Disease dataset.

## Setup
```bash
pip install -r requirements.txt
```
Download `heart_disease_uci.csv` from
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
and put it in `data/heart_disease_uci.csv` (overwrite the synthetic placeholder if present).

## Run
```bash
python -m src.train        # CSV -> SQLite -> tune -> evaluate -> save model + reports/
streamlit run app.py
```

## Layout
- `src/data.py` – cleaning, SQLite build, SQL queries
- `src/train.py` – pipeline, RandomizedSearchCV (5-fold, ROC-AUC), hold-out evaluation, plots
- `app.py` – Predict / Model evaluation / SQL explorer tabs
- `src/make_synthetic.py` – offline test data only (delete once you have the real CSV)

## Design notes
- Target = `num > 0` (any disease). `id` and `dataset` (hospital of origin) are excluded: not clinically meaningful and a leakage risk.
- Missing values are kept as NaN; XGBoost learns the best split direction. `chol == 0` / `trestbps == 0` are treated as missing.
- Threshold is chosen on out-of-fold *training* predictions (Youden's J), never on the test set.
- Test set is small (~180), so read AUC with its bootstrap CI.
