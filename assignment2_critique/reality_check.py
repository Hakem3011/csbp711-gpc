"""
CSBP 711 - Paper Critique reality check
Paper: Hassan et al. (2026) "Novel hybrid machine learning framework for high-fidelity prediction of fly ash-based
       geopolymer concrete strength", Composite Structures 378, 119906 (T-BoostNet).
Data:  the Assignment 1 data (Jiang et al. 2024, PLOS ONE S1 Table), cleaned exactly as in run_jiang.py (1,122 rows).

Configurations
  paper             the 17 inputs of the paper's SHAP plot (Fig. 13); target = converted (cube-equivalent) strength, the
                    range the paper reports (1.1-90.9 MPa); log(1+y) target (Eq. 2); median imputation and robust
                    scaling (Eq. 1) fitted inside every training fold
  paper_no_density  the same without density, which is measured on hardened concrete (an outcome, not a design input)
  a1                the Assignment 1 setup: 17 raw mix variables + specimen type, raw fc, median imputation,
                    standardisation and one-hot (run_jiang.pipeline), no target transform
Splits (5 folds, out-of-fold predictions)
  random    KFold(shuffle=True, seed)               the paper's protocol (Section 5.1)
  by_mix    GroupKFold on mix identity (shuffled)   no mix appears on both sides at different ages
  by_study  GroupKFold(5) on the reference          the same folds as Assignment 1
Models
  xgboost_paper      the paper's tuned XGBoost baseline (Table 3: 200 trees, depth 7, lr 0.1, reg_lambda 10)
  xgboost_a1         the Assignment 1 XGBoost (800 trees, depth 4, lr 0.04, subsample 0.9, colsample 0.9)
  tboostnet_literal  T-BoostNet as Eqs. (4)-(10) are written (see tboostnet.py)
  tboostnet_tokens   T-BoostNet with feature tokens (the generous reading)
  xgboost_stage1     control the paper does not report: T-BoostNet's own stage-1 XGBoost alone (200, depth 6, lr 0.1,
                     reg_lambda 10), i.e. T-BoostNet without its Transformer

Usage:  python reality_check.py --xlsx data/pone.0310422.s002.xlsx --seed 42
        optional filters: --configs paper a1 --splits random by_study --models xgboost_paper tboostnet_literal
Writes: results/reality_check.csv        pooled out-of-fold metrics, plus mean and sd over the five folds
        results/reality_check_folds.csv  one row per fold
"""
import argparse, os, re, time, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run_jiang.py is in the repository root
import run_jiang as rj
from tboostnet import TBoostNet
warnings.filterwarnings("ignore")

# the paper's 17 inputs (Fig. 13) mapped to the columns of the Jiang et al. sheet
PAPER_FEATURES = {
    "CT": "Heat-curing temperature (℃)", "Age": "Total curing period (day)",
    "W in AS": "H2O in alkaline solution (kg/m3)", "CP": "Heat-curing period (day)", "FA": "Fly ash (kg/m3)",
    "SH Molarity": "NaOH solution concentration (mol/L)", "CAgg": "Coarse aggregate (kg/m3)",
    "SS": "Na2SiO3 solution (kg/m3)", "SiO2 in AS": "SiO2 in alkaline solution (kg/m3)", "rho": "Density (kg/m3)",
    "W": "Extra added H2O (kg/m3)", "Na2O in AS": "Na2O in alkaline solution (kg/m3)", "SHS": "NaOH solution (kg/m3)",
    "FAgg": "Fine aggregate (kg/m3)", "Size": "Size_mm", "AS": "AS (kg/m3)", "Shape": "Shape_cylinder"}
CONVERTED = "Converted fc (Mpa)"
CONFIGS = ["paper", "paper_no_density", "a1"]
SPLITS = ["random", "by_mix", "by_study"]
MODELS = ["xgboost_paper", "xgboost_a1", "tboostnet_literal", "tboostnet_tokens", "xgboost_stage1"]


def prepare(path):
    df = rj.load(path, [])
    for c in ["H2O in alkaline solution (kg/m3)", "SiO2 in alkaline solution (kg/m3)", "Density (kg/m3)",
              "Na2O in alkaline solution (kg/m3)", CONVERTED]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Size_mm"] = df["Size"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)   # cube edge / cylinder diameter
    df["Shape_cylinder"] = (df["Specimens"] == "Cylinder").astype(float)
    df["AS (kg/m3)"] = df["NaOH solution (kg/m3)"] + df["Na2SiO3 solution (kg/m3)"]               # alkaline solution = SHS + SS
    # mix identity: same study, same proportions, chemistry and curing; only the test age (and strength) differ
    mixcols = [rj.GROUP] + [c for c in rj.RAW_FEATURES if c != "Total curing period (day)"] + rj.CATEGORICAL + ["Size"]
    key = df[mixcols].fillna("NA").astype(str).apply(lambda r: "|".join(r.tolist()), axis=1)
    df["mix_id"] = pd.factorize(key)[0]
    return df


def design(config, df):
    """inputs, target, target transform and preprocessing for one configuration"""
    if config in ("paper", "paper_no_density"):
        feats = [v for k, v in PAPER_FEATURES.items() if not (config == "paper_no_density" and k == "rho")]
        pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())])
        return df[feats], df[CONVERTED].values, True, (lambda m: Pipeline([("pre", clone(pre)), ("model", m)]))
    return (df[rj.RAW_FEATURES + rj.CATEGORICAL], df[rj.TARGET].values, False,
            (lambda m: rj.pipeline(m, rj.RAW_FEATURES, rj.CATEGORICAL)))


def splits(split, df, seed):
    idx = np.arange(len(df))
    if split == "random":
        return list(KFold(5, shuffle=True, random_state=seed).split(idx)), None
    if split == "by_mix":
        return list(GroupKFold(5, shuffle=True, random_state=seed).split(idx, groups=df["mix_id"])), df["mix_id"].values
    return list(GroupKFold(5).split(idx, groups=df[rj.GROUP])), df[rj.GROUP].values


def models(seed):
    return {"xgboost_paper": XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, reg_lambda=10,
                                          random_state=seed, n_jobs=2),
            "xgboost_a1": rj.models(seed)["xgboost"],
            "tboostnet_literal": TBoostNet("literal", seed=seed),
            "tboostnet_tokens": TBoostNet("tokens", seed=seed),
            # control the paper does not report: T-BoostNet's own stage-1 trees alone (Section 4.7 settings)
            "xgboost_stage1": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, reg_lambda=10,
                                           random_state=seed, n_jobs=2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="+", default=CONFIGS); ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--tag", default="")
    a = ap.parse_args(); os.makedirs("results", exist_ok=True)
    df = prepare(a.xlsx)
    print(f"rows {len(df)}; studies {df[rj.GROUP].nunique()}; distinct mixes {df['mix_id'].nunique()}", flush=True)
    out_path, fold_path = f"results/reality_check{a.tag}.csv", f"results/reality_check_folds{a.tag}.csv"
    for config in a.configs:
        X, y, log_target, make = design(config, df)
        for split in a.splits:
            folds, groups = splits(split, df, a.seed)
            for name in a.models:
                pred, frows, t, extra = np.zeros(len(y)), [], 0.0, ""
                for k, (tr, te) in enumerate(folds):
                    p = make(clone(models(a.seed)[name]))
                    yt = np.log1p(y[tr]) if log_target else y[tr]
                    fit_kw = {"model__groups": groups[tr]} if (name.startswith("tboostnet") and groups is not None) else {}
                    t0 = time.perf_counter(); p.fit(X.iloc[tr], yt, **fit_kw); t += time.perf_counter() - t0
                    yp = p.predict(X.iloc[te]); yp = np.expm1(yp) if log_target else yp
                    pred[te] = yp
                    m = p.named_steps["model"]
                    frows.append({"config": config, "split": split, "model": name, "fold": k,
                                  "R2": r2_score(y[te], yp), "MAE": mean_absolute_error(y[te], yp),
                                  "RMSE": mean_squared_error(y[te], yp) ** 0.5,
                                  "epochs": getattr(m, "epochs_run_", np.nan)})
                fr = pd.DataFrame(frows)
                row = {"config": config, "split": split, "model": name, "seed": a.seed,
                       "R2_pooled": r2_score(y, pred), "MAE_pooled": mean_absolute_error(y, pred),
                       "RMSE_pooled": mean_squared_error(y, pred) ** 0.5,
                       "R2_fold_mean": fr.R2.mean(), "R2_fold_sd": fr.R2.std(ddof=0),
                       "MAE_fold_mean": fr.MAE.mean(), "MAE_fold_sd": fr.MAE.std(ddof=0), "fit_time_s": t}
                pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
                fr.to_csv(fold_path, mode="a", header=not os.path.exists(fold_path), index=False)
                print(f"{config:16s} {split:8s} {name:18s} R2 {row['R2_pooled']:.3f} (folds {row['R2_fold_mean']:.3f} "
                      f"+/- {row['R2_fold_sd']:.3f})  MAE {row['MAE_pooled']:.2f}  {t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
