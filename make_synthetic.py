"""Synthetic stand-in with the SAME schema as the Kaggle file (offline testing only)."""
import numpy as np, pandas as pd
from .config import CSV_PATH


def make(n=920, seed=0):
    r = np.random.default_rng(seed)
    age = r.integers(29, 78, n)
    sex = r.choice(["Male", "Female"], n, p=[.79, .21])
    cp = r.choice(["typical angina", "atypical angina", "non-anginal", "asymptomatic"], n, p=[.05, .19, .22, .54])
    trestbps = r.normal(132, 19, n).round()
    chol = r.normal(200, 110, n).clip(0, 600).round()
    fbs = r.choice([True, False], n, p=[.17, .83])
    restecg = r.choice(["normal", "st-t abnormality", "lv hypertrophy"], n, p=[.6, .2, .2])
    thalch = (205 - age * .75 + r.normal(0, 20, n)).round()
    exang = r.choice([True, False], n, p=[.4, .6])
    oldpeak = r.gamma(1.2, .8, n).round(1)
    slope = r.choice(["upsloping", "flat", "downsloping"], n, p=[.2, .6, .2])
    ca = r.choice([0, 1, 2, 3], n, p=[.6, .2, .13, .07]).astype(float)
    thal = r.choice(["normal", "fixed defect", "reversable defect"], n, p=[.5, .1, .4])
    z = (-5.0 + .035 * age + 1.0 * (sex == "Male") + 1.4 * (cp == "asymptomatic") + .9 * exang
         + .6 * oldpeak + .6 * ca + .9 * (thal == "reversable defect") - .015 * (thalch - 140)
         + 1.0 * (slope == "flat") + r.normal(0, .8, n))
    p = 1 / (1 + np.exp(-z))
    num = np.where(r.random(n) < p, r.choice([1, 2, 3, 4], n, p=[.45, .27, .18, .10]), 0)
    df = pd.DataFrame(dict(id=np.arange(1, n + 1), age=age, sex=sex,
        dataset=r.choice(["Cleveland", "Hungary", "Switzerland", "VA Long Beach"], n),
        cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalch=thalch,
        exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal, num=num))
    for c, f in [("ca", .66), ("thal", .53), ("slope", .33), ("chol", .03), ("fbs", .1),
                 ("trestbps", .06), ("thalch", .06), ("exang", .06), ("oldpeak", .07), ("restecg", .002)]:
        df[c] = df[c].astype(object)
        df.loc[r.random(n) < f, c] = np.nan
    return df


if __name__ == "__main__":
    make().to_csv(CSV_PATH, index=False)
    print("wrote", CSV_PATH)
