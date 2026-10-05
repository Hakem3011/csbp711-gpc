"""
Ablation 5 - is XGBoost's margin over the forest due to boosting (fitting residuals) or to capacity/implementation?
Prediction, stated before running: if residual fitting is the mechanism, XGBRegressor (boosted) should beat
XGBRFRegressor (bagged, same library, same depth 4, same subsample and column sampling) on the by-study split.
If the two are level, the margin is not about boosting and the gap stays open.

Usage:  python ablation_boosting.py --xlsx data/pone.0310422.s002.xlsx --seed 42
Writes: results/ablation_boosting.csv
"""
import argparse, os, warnings, numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor, XGBRFRegressor
import run_jiang as rj
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", required=True); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); os.makedirs("results", exist_ok=True)
df = rj.load(a.xlsx, []); X = df[rj.RAW_FEATURES + rj.CATEGORICAL]; y = df[rj.TARGET].values
common = dict(n_estimators=800, max_depth=4, subsample=0.9, random_state=a.seed, n_jobs=2)
models = {
    "xgb_boosted_depth4": XGBRegressor(learning_rate=0.04, colsample_bytree=0.9, **common),          # as in run_jiang.py
    "xgb_bagged_depth4":  XGBRFRegressor(colsample_bynode=0.9, **common),                             # same library, no boosting
}
rows = []
for split in ["random", "group"]:
    folds = list(KFold(5, shuffle=True, random_state=a.seed).split(X)) if split == "random" else list(GroupKFold(5).split(X, y, groups=df[rj.GROUP]))
    for name, m in models.items():
        pred = np.zeros(len(y))
        for tr, te in folds:
            p = rj.pipeline(m, rj.RAW_FEATURES, rj.CATEGORICAL); p.fit(X.iloc[tr], y[tr]); pred[te] = p.predict(X.iloc[te])
        rows.append({"split": split, "model": name, "MAE_MPa": mean_absolute_error(y, pred), "R2": r2_score(y, pred)})
out = pd.DataFrame(rows); out.to_csv("results/ablation_boosting.csv", index=False)
print(out.round(3).to_string(index=False))
