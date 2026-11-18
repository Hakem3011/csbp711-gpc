"""
CSBP 711 - Paper Critique: robustness of the by-study result over re-drawn fold assignments
(same idea as ablation_folds.py in Assignment 1: GroupKFold with shuffle, 8 draws of which studies share a fold).

Usage:  python reality_check_draws.py --xlsx data/pone.0310422.s002.xlsx --draws 8
Writes: results/reality_check_draws.csv (one row per draw, configuration and model) and prints a summary
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run_jiang.py is in the repository root
import run_jiang as rj
import reality_check as rc
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--configs", nargs="+", default=["paper", "a1"])
ap.add_argument("--models", nargs="+", default=["xgboost_paper", "xgboost_stage1", "xgboost_a1", "tboostnet_literal"])
a = ap.parse_args(); os.makedirs("results", exist_ok=True)
df = rc.prepare(a.xlsx); groups = df[rj.GROUP].values
path = "results/reality_check_draws.csv"
for d in range(a.draws):
    folds = list(GroupKFold(5, shuffle=True, random_state=d).split(np.arange(len(df)), groups=groups))
    for config in a.configs:
        X, y, log_target, make = rc.design(config, df)
        for name in a.models:
            pred = np.zeros(len(y))
            for tr, te in folds:
                p = make(clone(rc.models(a.seed)[name]))
                fit_kw = {"model__groups": groups[tr]} if name.startswith("tboostnet") else {}
                p.fit(X.iloc[tr], np.log1p(y[tr]) if log_target else y[tr], **fit_kw)
                yp = p.predict(X.iloc[te]); pred[te] = np.expm1(yp) if log_target else yp
            row = {"draw": d, "config": config, "model": name, "R2_by_study": r2_score(y, pred),
                   "MAE_by_study": mean_absolute_error(y, pred)}
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"draw {d} {config:6s} {name:18s} R2 {row['R2_by_study']:.3f}", flush=True)
out = pd.read_csv(path)
print("\nby-study R2 over", out.draw.nunique(), "draws:\n",
      out.groupby(["config", "model"])["R2_by_study"].agg(["mean", "std", "min", "max"]).round(3).to_string())
for config in out.config.unique():
    w = out[out.config == config].pivot(index="draw", columns="model", values="R2_by_study")
    if {"tboostnet_literal", "xgboost_a1"} <= set(w.columns):
        print(f"{config}: T-BoostNet above the Assignment 1 XGBoost in {(w['tboostnet_literal'] > w['xgboost_a1']).sum()} of {len(w)} draws")
    if {"tboostnet_literal", "xgboost_stage1"} <= set(w.columns):
        print(f"{config}: T-BoostNet above its own stage-1 XGBoost in {(w['tboostnet_literal'] > w['xgboost_stage1']).sum()} of {len(w)} draws")
