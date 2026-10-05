"""
Ablation 4 - how much do the by-study results depend on which studies share a fold?
GroupKFold assigns the 83 reference blocks to folds deterministically, so a fixed seed does not vary them.
This script re-draws the assignment 8 times (GroupKFold with shuffle, scikit-learn >= 1.6) and reports
mean and standard deviation of the pooled out-of-fold R2 for each model, plus how often each ranking held.

Usage:  python ablation_folds.py --xlsx data/pone.0310422.s002.xlsx --draws 8
Writes: results/ablation_folds.csv   (one row per draw and model) and prints the summary.
Runtime: about 1 minute per draw on two cores.
"""
import argparse, os, warnings, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import run_jiang as rj
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", required=True); ap.add_argument("--draws", type=int, default=8)
a = ap.parse_args(); os.makedirs("results", exist_ok=True)
df = rj.load(a.xlsx, []); X = df[rj.RAW_FEATURES + rj.CATEGORICAL]; y = df[rj.TARGET].values
rows = []
for d in range(a.draws):
    try:
        folds = list(GroupKFold(5, shuffle=True, random_state=d).split(X, y, groups=df[rj.GROUP]))
    except TypeError:
        raise SystemExit("This scikit-learn is older than 1.6 and cannot shuffle GroupKFold; upgrade with: pip install -U scikit-learn")
    for name, model in rj.models(42).items():
        pred = np.zeros(len(y))
        for tr, te in folds:
            p = rj.pipeline(model, rj.RAW_FEATURES, rj.CATEGORICAL); p.fit(X.iloc[tr], y[tr]); pred[te] = p.predict(X.iloc[te])
        rows.append({"draw": d, "model": name, "R2_by_study": r2_score(y, pred)})
    print(f"draw {d} done")
out = pd.DataFrame(rows); out.to_csv("results/ablation_folds.csv", index=False)
summary = out.groupby("model")["R2_by_study"].agg(["mean", "std", "min", "max"]).round(3)
print("\nby-study R2 over", a.draws, "fold draws:\n", summary.to_string())
wide = out.pivot(index="draw", columns="model", values="R2_by_study")
print("\nXGBoost > forest in", int((wide["xgboost"] > wide["random_forest"]).sum()), "of", a.draws, "draws;",
      "forest > ridge in", int((wide["random_forest"] > wide["ridge_baseline"]).sum()), "of", a.draws, ";",
      "MLP last in", int((wide["mlp"] == wide.min(axis=1)).sum()), "of", a.draws)
