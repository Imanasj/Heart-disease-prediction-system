"""Train + evaluate the XGBoost heart-disease model.   Run:  python -m src.train"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve, classification_report)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from . import config as C
from .data import load_and_clean, build_database, load_training_frame


def build_pipeline(scale_pos_weight=1.0) -> Pipeline:
    pre = ColumnTransformer([
        # XGBoost handles NaN natively, so numeric columns are passed through untouched
        ("num", "passthrough", C.NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), C.CATEGORICAL),
    ])
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                        scale_pos_weight=scale_pos_weight, random_state=C.RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def tidy_categoricals(df):
    df = df.copy()
    for c in C.CATEGORICAL:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def main():
    # 1) CSV -> clean -> SQLite -> SQL query -> DataFrame
    build_database(load_and_clean())
    df = tidy_categoricals(load_training_frame())
    X, y = df[C.FEATURES], df[C.TARGET]
    print(f"Rows: {len(df)} | positive rate: {y.mean():.2%}")

    # 2) Hold-out test set (never touched while tuning)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

    # 3) Hyper-parameter search with stratified CV on the training split only
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    space = {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_lambda": [1, 5, 10],
        "clf__gamma": [0, 0.1, 0.5],
    }
    search = RandomizedSearchCV(build_pipeline(spw), space, n_iter=40, scoring="roc_auc", cv=cv,
                                random_state=C.RANDOM_STATE, n_jobs=-1, refit=True, verbose=0)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_
    print("Best CV ROC-AUC:", round(search.best_score_, 4))
    print("Best params:", search.best_params_)

    # 4) Evaluate on the hold-out set
    proba = model.predict_proba(X_te)[:, 1]
    # Choose the decision threshold on TRAIN out-of-fold predictions (never on test)
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    fpr_o, tpr_o, thr_o = roc_curve(y_tr, oof)
    threshold = float(thr_o[np.argmax(tpr_o - fpr_o)])            # Youden's J: balances sensitivity & specificity
    pred = (proba >= threshold).astype(int)
    pred_05 = (proba >= 0.5).astype(int)

    def block(p):
        tn, fp, fn, tp = confusion_matrix(y_te, p).ravel()
        return dict(accuracy=accuracy_score(y_te, p), precision=precision_score(y_te, p),
                    recall=recall_score(y_te, p), specificity=tn / (tn + fp), f1=f1_score(y_te, p),
                    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    # Bootstrap 95% CI for AUC (small test set -> wide uncertainty is normal)
    rng = np.random.default_rng(C.RANDOM_STATE)
    yt = y_te.to_numpy()
    aucs = []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        if len(set(yt[i])) == 2:
            aucs.append(roc_auc_score(yt[i], proba[i]))

    metrics = dict(
        n_train=len(X_tr), n_test=len(X_te), positive_rate=float(y.mean()),
        cv_roc_auc=float(search.best_score_), best_params={k.split("__")[1]: v for k, v in search.best_params_.items()},
        test_roc_auc=roc_auc_score(y_te, proba), test_roc_auc_ci=[float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        test_pr_auc=average_precision_score(y_te, proba), brier=brier_score_loss(y_te, proba),
        threshold=threshold, at_threshold=block(pred), at_0_5=block(pred_05),
    )
    print(classification_report(y_te, pred, target_names=["No disease", "Disease"]))
    print(f"Test ROC-AUC {metrics['test_roc_auc']:.3f}  CI {metrics['test_roc_auc_ci']}  threshold {threshold:.2f}")

    # 5) Plots
    C.REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fpr, tpr, _ = roc_curve(y_te, proba)
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC={metrics['test_roc_auc']:.3f}"); ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend()
    p_, r_, _ = precision_recall_curve(y_te, proba)
    ax[1].plot(r_, p_, lw=2, label=f"AP={metrics['test_pr_auc']:.3f}")
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision"); ax[1].legend()
    cm = confusion_matrix(y_te, pred)
    ax[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax[2].text(j, i, v, ha="center", va="center", fontsize=16)
    ax[2].set(title=f"Confusion matrix (thr={threshold:.2f})", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], xlabel="Predicted", ylabel="Actual")
    plt.tight_layout(); plt.savefig(C.REPORTS / "evaluation.png", dpi=130); plt.close()

    names = model.named_steps["pre"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values().tail(15)
    imp.plot.barh(figsize=(7, 5), title="Top feature importances (gain-based)")
    plt.tight_layout(); plt.savefig(C.REPORTS / "feature_importance.png", dpi=130); plt.close()
    imp.sort_values(ascending=False).to_csv(C.REPORTS / "feature_importance.csv", header=["importance"])

    # 6) Persist
    joblib.dump({"model": model, "threshold": threshold, "features": C.FEATURES}, C.MODEL_PATH)
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print("Saved:", C.MODEL_PATH)


if __name__ == "__main__":
    main()
