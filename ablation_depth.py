"""
Ablation 3 - does tree depth explain why XGBoost beats the random forest across laboratories?
Prediction (stated before running): limiting the forest to depth 4 (XGBoost's depth) should raise its
by-study R2 towards XGBoost's, because deep trees fit each laboratory's constant columns.

Usage: python ablation_depth.py --xlsx data/pone.0310422.s002.xlsx --seed 42
Writes: results/ablation_depth.csv
"""
import argparse, os, warnings, numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import run_jiang as rj
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", required=True); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
df = rj.load(a.xlsx, [])
X = df[rj.RAW_FEATURES + rj.CATEGORICAL]; y = df[rj.TARGET].values
os.makedirs("results", exist_ok=True)
rows = []
for split in ["random", "group"]:
    folds = list(KFold(5, shuffle=True, random_state=a.seed).split(X)) if split == "random" else list(GroupKFold(5).split(X, y, groups=df[rj.GROUP]))
    for depth in [None, 8, 4]:
        pred = np.zeros(len(y)); size = 0
        for tr, te in folds:
            p = rj.pipeline(RandomForestRegressor(n_estimators=500, max_depth=depth, min_samples_leaf=2, random_state=a.seed, n_jobs=2), rj.RAW_FEATURES, rj.CATEGORICAL)
            p.fit(X.iloc[tr], y[tr]); pred[te] = p.predict(X.iloc[te]); size = rj.size_of(p)
        rows.append({"split": split, "model": f"random_forest_depth_{depth or 'unlimited'}", "MAE_MPa": mean_absolute_error(y, pred), "R2": r2_score(y, pred), "model_size": size})
out = pd.DataFrame(rows); out.to_csv("results/ablation_depth.csv", index=False)
print(out.round(3).to_string(index=False))
