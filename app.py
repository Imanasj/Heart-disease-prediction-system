"""Streamlit app.  Run:  streamlit run app.py"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config as C
from data import query, DB_PATH

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(C.MODEL_PATH)


def bootstrap():
    """First launch: make sure DB + model exist. Trains automatically if the CSV is present."""
    if not C.CSV_PATH.exists():
        st.warning("Dataset not found. Upload `heart_disease_uci.csv` from Kaggle "
                   "(kaggle.com/datasets/redwankarimsony/heart-disease-data) to get started.")
        up = st.file_uploader("heart_disease_uci.csv", type="csv")
        if up is None:
            st.stop()
        C.CSV_PATH.write_bytes(up.getvalue())
        st.rerun()
    if not C.DB_PATH.exists():
        from data import load_and_clean, build_database
        build_database(load_and_clean())
    if not C.MODEL_PATH.exists() or not C.METRICS_PATH.exists():
        import train
        with st.spinner("First launch: training the model (about a minute)..."):
            train.main(n_iter=25)
        st.rerun()


if not (C.MODEL_PATH.exists() and C.DB_PATH.exists() and C.METRICS_PATH.exists()):
    bootstrap()

bundle = load_model()
model, default_thr = bundle["model"], bundle["threshold"]
metrics = json.loads(C.METRICS_PATH.read_text())

st.title("❤️ Heart Disease Prediction System")
st.caption("XGBoost model trained on the UCI Heart Disease dataset. Educational demo — not a medical device.")
tab_pred, tab_eval, tab_data = st.tabs(["🩺 Predict", "📈 Model evaluation", "🗄️ Data explorer (SQL)"])

# ---------------------------------------------------------------- Predict
with tab_pred:
    st.sidebar.header("Decision threshold")
    thr = st.sidebar.slider("Flag as 'disease' when risk ≥", 0.05, 0.95, float(round(default_thr, 2)), 0.01,
                            help="Lower = catch more cases (higher recall) but more false alarms.")
    st.sidebar.caption(f"Tuned default: {default_thr:.2f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 20, 100, 55)
        sex = st.selectbox("Sex", ["male", "female"])
        cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.number_input("Resting blood pressure (mmHg)", 80, 220, 130)
    with c2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["false", "true"])
        restecg = st.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.number_input("Max heart rate achieved", 60, 220, 150)
    with c3:
        exang = st.selectbox("Exercise-induced angina", ["false", "true"])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("ST slope", ["upsloping", "flat", "downsloping"])
        ca_known = st.checkbox("Fluoroscopy vessel count is known", value=True)
        ca = st.slider("Major vessels colored (0-3)", 0, 3, 0, disabled=not ca_known)
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    if st.button("Predict risk", type="primary"):
        row = pd.DataFrame([dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
                                 thalch=thalch, exang=exang, oldpeak=oldpeak, slope=slope,
                                 ca=ca if ca_known else np.nan, thal=thal)])[C.FEATURES]
        risk = float(model.predict_proba(row)[0, 1])
        label = int(risk >= thr)
        a, b = st.columns([1, 2])
        a.metric("Estimated risk", f"{risk:.0%}")
        a.progress(min(max(risk, 0.0), 1.0))
        (b.error if label else b.success)(
            "⚠️ Elevated risk — flagged for clinical follow-up." if label else "✅ Below the decision threshold.")
        b.caption("This is a statistical estimate from a small, older dataset. It cannot diagnose anything.")
        # Log to SQL
        import sqlite3
        try:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("INSERT INTO predictions (age, sex, cp, trestbps, chol, risk, label) VALUES (?,?,?,?,?,?,?)",
                            (age, sex, cp, trestbps, chol, risk, label))
        except sqlite3.Error:
            pass

# ---------------------------------------------------------------- Evaluation
with tab_eval:
    m, t = metrics, metrics["at_threshold"]
    k = st.columns(5)
    k[0].metric("Test ROC-AUC", f"{m['test_roc_auc']:.3f}",
                help=f"95% bootstrap CI: {m['test_roc_auc_ci'][0]:.2f}–{m['test_roc_auc_ci'][1]:.2f}")
    k[1].metric("CV ROC-AUC", f"{m['cv_roc_auc']:.3f}")
    k[2].metric("Recall", f"{t['recall']:.2f}")
    k[3].metric("Precision", f"{t['precision']:.2f}")
    k[4].metric("Specificity", f"{t['specificity']:.2f}")
    st.caption(f"Hold-out test set: {m['n_test']} patients · threshold {m['threshold']:.2f} · "
               f"95% CI for AUC {m['test_roc_auc_ci'][0]:.2f}–{m['test_roc_auc_ci'][1]:.2f}")
    st.image(str(C.REPORTS / "evaluation.png"))
    left, right = st.columns(2)
    left.image(str(C.REPORTS / "feature_importance.png"))
    right.subheader("Threshold comparison")
    right.dataframe(pd.DataFrame({"0.50": m["at_0_5"], f"tuned ({m['threshold']:.2f})": m["at_threshold"]}).round(3))
    with st.expander("Best hyper-parameters"):
        st.json(m["best_params"])

# ---------------------------------------------------------------- SQL explorer
with tab_data:
    st.write("Query the `patients` and `predictions` tables (read-only SELECT).")
    presets = {
        "Disease rate by chest pain type":
            "SELECT cp, COUNT(*) AS n, ROUND(AVG(target)*100,1) AS pct_disease\nFROM patients GROUP BY cp ORDER BY pct_disease DESC",
        "Disease rate by age band & sex":
            "SELECT (age/10)*10 AS age_band, sex, COUNT(*) AS n, ROUND(AVG(target)*100,1) AS pct_disease\nFROM patients GROUP BY age_band, sex ORDER BY age_band, sex",
        "Missing values per column":
            "SELECT SUM(ca IS NULL) AS ca, SUM(thal IS NULL) AS thal, SUM(slope IS NULL) AS slope,\n       SUM(chol IS NULL) AS chol, SUM(fbs IS NULL) AS fbs, COUNT(*) AS total FROM patients",
        "Recent predictions": "SELECT * FROM predictions ORDER BY ts DESC LIMIT 20",
    }
    choice = st.selectbox("Preset", list(presets))
    sql = st.text_area("SQL", presets[choice], height=120)
    if st.button("Run query"):
        if not sql.strip().lower().startswith("select") or ";" in sql.strip().rstrip(";"):
            st.warning("Only a single SELECT statement is allowed.")
        else:
            try:
                res = query(sql)
                st.dataframe(res, width="stretch")
                if res.shape[1] == 2 and pd.api.types.is_numeric_dtype(res.iloc[:, 1]):
                    st.bar_chart(res.set_index(res.columns[0]))
            except Exception as e:
                st.error(f"Query failed: {e}")
