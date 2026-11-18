"""
CSBP 711 - How much of the error on a new lab is one shift per lab?

For held-out-study predictions (the 8 re-drawn GroupKFold(5) assignments of reality_check_draws.py),
split each residual into its study's mean residual (the lab offset) and the rest, and report
- R2_by_study       pooled out-of-fold R2 (should match results/reality_check_draws.csv)
- offset_share_raw  sum_s n_s * mean_resid_s^2 / sum resid^2   (overstated by studies with very few rows)
- icc               one-way random-effects ANOVA share of the residual variance that lies between studies
- R2_oracle_offset  R2 if each held-out study's mean residual were known: the ceiling for offset calibration

Usage:  python offset_share.py --xlsx data/pone.0310422.s002.xlsx
Writes: results/offset_share.csv
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run_jiang.py is in the repository root
import run_jiang as rj
import reality_check as rc
warnings.filterwarnings("ignore")


def icc_oneway(r, g):
    d = pd.DataFrame({"r": r, "g": g})
    grp = d.groupby("g")["r"]
    n, m = grp.size().values.astype(float), grp.mean().values
    N, S = len(d), len(n)
    ssb = (n * (m - d.r.mean()) ** 2).sum()
    ssw = ((d.r - grp.transform("mean")) ** 2).sum()
    msb, msw = ssb / (S - 1), ssw / (N - S)
    n0 = (N - (n ** 2).sum() / N) / (S - 1)
    sb2 = max((msb - msw) / n0, 0.0)
    return sb2 / (sb2 + msw)


ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--pairs", nargs="+", default=["a1:xgboost_a1", "paper:xgboost_paper"])
a = ap.parse_args(); os.makedirs("results", exist_ok=True)
df = rc.prepare(a.xlsx); groups = df[rj.GROUP].values
rows = []
for d in range(a.draws):
    folds = list(GroupKFold(5, shuffle=True, random_state=d).split(np.arange(len(df)), groups=groups))
    for pair in a.pairs:
        config, name = pair.split(":")
        X, y, log_target, make = rc.design(config, df)
        pred = np.zeros(len(y))
        for tr, te in folds:
            p = make(clone(rc.models(a.seed)[name]))
            p.fit(X.iloc[tr], np.log1p(y[tr]) if log_target else y[tr])
            yp = p.predict(X.iloc[te]); pred[te] = np.expm1(yp) if log_target else yp
        r = y - pred
        off = pd.Series(r).groupby(groups).transform("mean").values
        rows.append({"draw": d, "config": config, "model": name, "R2_by_study": r2_score(y, pred),
                     "offset_share_raw": (off ** 2).sum() / (r ** 2).sum(), "icc": icc_oneway(r, groups),
                     "R2_oracle_offset": 1 - ((r - off) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
out = pd.DataFrame(rows); out.to_csv("results/offset_share.csv", index=False)
cols = ["R2_by_study", "offset_share_raw", "icc", "R2_oracle_offset"]
print("\n" + out.groupby(["config", "model"])[cols].agg(["mean", "std"]).round(3).to_string())
n = pd.Series(groups).value_counts()
print(f"\nstudies {len(n)}; rows per study: median {n.median():.0f}, min {n.min()}, max {n.max()}; "
      f"studies with 1 row: {(n == 1).sum()}; with <= 3 rows: {(n <= 3).sum()}")
