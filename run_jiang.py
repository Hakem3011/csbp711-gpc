"""
CSBP 711 - Assignment 1
Dataset: Jiang et al. (2024) PLOS ONE, doi 10.1371/journal.pone.0310422, S1 Table = file pone.0310422.s002.xlsx
         (1,136 fly-ash geopolymer concrete mixes in 83 reference blocks from 82 papers, CC BY 4.0).

Usage:  python run_jiang.py --xlsx data/pone.0310422.s002.xlsx --seed 42
Writes: results/data_audit.txt, results/comparison.csv, results/ablation.csv, results/run_config.json, figures/*.png
"""
import argparse, os, re, time, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor; HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

RAW_FEATURES = ["SiO2 in fly ash (%)", "Al2O3 in fly ash (%)", "CaO in fly ash (%)", "Fe2O3 in fly ash (%)",
                "MgO in fly ash (%)", "Fly ash (kg/m3)", "NaOH solution (kg/m3)", "NaOH solution concentration (mol/L)",
                "Na2SiO3 solution (kg/m3)", "Na2O in Na2SiO3 solution (%)", "SiO2 in Na2SiO3 solution (%)",
                "Extra added H2O (kg/m3)", "Coarse aggregate (kg/m3)", "Fine aggregate (kg/m3)",
                "Heat-curing temperature (℃)", "Heat-curing period (day)", "Total curing period (day)"]
CATEGORICAL = ["Specimens"]          # cube / cylinder, kept as a feature instead of applying the authors' fixed factors (x1.18 / x1.04)
TARGET = "fc (Mpa)"
GROUP = "Reference"
DROPPED_SPARSE = ["NaOH solids (kg/m3)", "H2O in NaOH solution (kg/m3)"]   # 79 % missing: only some papers report them

# ------------------------------------------------------------------ data
def load(path, log):
    df = pd.read_excel(path)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    n0 = len(df)
    log.append(f"rows read: {n0}; columns: {df.shape[1]}")
    # block-sparse columns (no merged cells in the workbook): the reference and the specimen size are written on the
    # first row of each study block only, so they are forward-filled; fly-ash type is a within-study index and is unused
    for c in [GROUP, "Size"]:
        blank = df[c].isna().sum(); df[c] = df[c].ffill()
        log.append(f"forward-filled block-sparse column '{c}': {blank} blank cells (value written once per study block)")
    stray = [c for c in df.columns if c.startswith("Unnamed")]
    log.append(f"stray unnamed columns in the sheet, ignored: {stray}")
    for c in RAW_FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    miss = df[RAW_FEATURES + DROPPED_SPARSE].isna().mean().sort_values(ascending=False)
    log.append("missing fraction, worst five columns: " + ", ".join(f"{k}={v:.2f}" for k, v in miss.head(5).items()))
    log.append(f"dropped sparse columns (not usable): {DROPPED_SPARSE}")
    dup = df.duplicated(subset=RAW_FEATURES + CATEGORICAL + [TARGET]).sum()
    df = df.drop_duplicates(subset=RAW_FEATURES + CATEGORICAL + [TARGET]).reset_index(drop=True)
    log.append(f"exact duplicate rows removed: {dup}; rows kept: {len(df)}")
    fp = df[["SiO2 in fly ash (%)", "Al2O3 in fly ash (%)", "CaO in fly ash (%)", "Fe2O3 in fly ash (%)"]].round(2).astype(str).agg("|".join, axis=1)
    shared = int((df.groupby(fp)[GROUP].nunique() > 1).sum())
    log.append(f"distinct fly-ash oxide fingerprints: {fp.nunique()}; fingerprints used by more than one reference: {shared} "
               f"(the by-study split holds out reference strings, not laboratories)")
    vc = df[GROUP].value_counts()
    log.append(f"source studies: {df[GROUP].nunique()}; rows per study median {int(vc.median())}, max {vc.max()} "
               f"({vc.max()/len(df):.0%} of all rows from one study)")
    log.append(f"target {TARGET}: mean {df[TARGET].mean():.1f}, min {df[TARGET].min():.1f}, max {df[TARGET].max():.1f} MPa")
    log.append("specimen types: " + ", ".join(f"{k}={v}" for k, v in df["Specimens"].value_counts().items()))
    return df

def between_study_share(df):
    """share of target variance that lies between studies (eta-squared of the study factor)"""
    grand = df[TARGET].mean(); tot = ((df[TARGET] - grand) ** 2).sum()
    means = df.groupby(GROUP)[TARGET].transform("mean"); between = ((means - grand) ** 2).sum()
    return between / tot

# ------------------------------------------------------------------ models
def models(seed):
    m = {"ridge_baseline": Ridge(alpha=1.0),
         "random_forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=2),
         "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=4000, random_state=seed)}
    if HAVE_XGB:
        m["xgboost"] = XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.04, subsample=0.9,
                                    colsample_bytree=0.9, random_state=seed, n_jobs=2)
    else:
        m["gradient_boosting"] = GradientBoostingRegressor(n_estimators=800, max_depth=3, learning_rate=0.04, random_state=seed)
    return m

def pipeline(model, numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    return Pipeline([("pre", pre), ("model", model)])

def size_of(pipe):
    m = pipe.named_steps["model"]
    if hasattr(m, "estimators_"): return int(sum(t.tree_.node_count for t in np.ravel(m.estimators_)))
    if hasattr(m, "get_booster"): return int(sum(len([l for l in t.split("\n") if l.strip()]) for t in m.get_booster().get_dump()))
    if hasattr(m, "coefs_"): return int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_))
    if hasattr(m, "coef_"): return int(np.size(m.coef_) + 1)
    return -1

def run(df, numeric, categorical, split, seed, label):
    X = df[numeric + categorical]; y = df[TARGET].values
    if split == "random": folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    else:                 folds = list(GroupKFold(5).split(X, y, groups=df[GROUP]))
    rows, oof = [], {}
    for name, model in models(seed).items():
        pred = np.zeros(len(y)); t = 0.0; size = 0
        for tr, te in folds:
            p = pipeline(model, numeric, categorical)
            t0 = time.perf_counter(); p.fit(X.iloc[tr], y[tr]); t += time.perf_counter() - t0
            pred[te] = p.predict(X.iloc[te]); size = size_of(p)
        rows.append({"setting": label, "split": split, "model": name, "MAE_MPa": mean_absolute_error(y, pred),
                     "R2": r2_score(y, pred), "fit_time_s": t, "model_size": size})
        oof[name] = pred
    return rows, oof

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", required=True); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    log = []; df = load(a.xlsx, log)
    share = between_study_share(df); log.append(f"share of strength variance BETWEEN studies: {share:.0%}")
    open("results/data_audit.txt", "w").write("\n".join(log)); print("\n".join(log))

    # main comparison: 4 models x 2 splits
    rows, oof_r = run(df, RAW_FEATURES, CATEGORICAL, "random", a.seed, "mix variables")
    rows2, oof_g = run(df, RAW_FEATURES, CATEGORICAL, "group", a.seed, "mix variables")
    comp = pd.DataFrame(rows + rows2); comp.to_csv("results/comparison.csv", index=False)
    print("\n", comp.round(3).to_string(index=False))

    # ablation 1: give every model the study identity (random split only; it is meaningless for unseen studies)
    df["study_id"] = df[GROUP].astype("category").cat.codes.astype(str)
    rows3, _ = run(df, RAW_FEATURES, CATEGORICAL + ["study_id"], "random", a.seed, "mix variables + study identity")
    # ablation 2: the 'know the lab' oracle - predict each row by the mean strength of its study in the training fold
    y = df[TARGET].values; pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=a.seed).split(df):
        means = df.iloc[tr].groupby(GROUP)[TARGET].mean(); grand = y[tr].mean()
        pred[te] = df.iloc[te][GROUP].map(means).fillna(grand).values
    rows3.append({"setting": "study-mean oracle (no mix variables)", "split": "random", "model": "study_mean",
                  "MAE_MPa": mean_absolute_error(y, pred), "R2": r2_score(y, pred), "fit_time_s": 0.0, "model_size": df[GROUP].nunique()})
    abl = pd.DataFrame(rows3); abl.to_csv("results/ablation.csv", index=False); print("\n", abl.round(3).to_string(index=False))

    # ---------------- figures
    # F1 data: strength distribution and rows per study
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(df[TARGET], bins=30, color="#3b6e8f"); ax[0].set_xlabel("Compressive strength (MPa)"); ax[0].set_ylabel("Mixes")
    vc = df[GROUP].value_counts().values; ax[1].bar(range(len(vc)), vc, color="#3b6e8f"); ax[1].set_xlabel("Study (sorted)"); ax[1].set_ylabel("Mixes per study")
    ax[1].set_title(f"{df[GROUP].nunique()} studies; largest = {vc.max()} mixes", fontsize=9)
    plt.tight_layout(); plt.savefig("figures/F1_data.png", dpi=200); plt.close()

    # F2 fingerprint: within-study spread of each feature as a fraction of the overall spread
    within = df.groupby(GROUP)[RAW_FEATURES].std().mean() / df[RAW_FEATURES].std()
    within = within.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2)); ax.barh([c.replace(" (kg/m3)", "").replace(" (%)", "") for c in within.index], within.values, color="#3b6e8f")
    ax.set_xlabel("Within-study std / overall std  (small = laboratory fingerprint)"); plt.tight_layout(); plt.savefig("figures/F2_fingerprint.png", dpi=200); plt.close()

    # F3 comparison: R2 by model under both splits
    piv = comp.pivot(index="model", columns="split", values="R2").loc[list(oof_r.keys())]
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(len(piv)); w = 0.38
    ax.bar(x - w/2, piv["random"], w, label="random split (leaky)", color="#9bb7cf"); ax.bar(x + w/2, piv["group"], w, label="split by source study", color="#1b2a4a")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel("R² (5-fold, out of fold)"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig("figures/F3_comparison.png", dpi=200); plt.close()

    # F4 residuals by study for the forest: random vs grouped split
    best = "random_forest"; fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, (title, pred) in enumerate([("random split", oof_r[best]), ("split by study", oof_g[best])]):
        res = pd.DataFrame({"res": y - pred, GROUP: df[GROUP]}); order = res.groupby(GROUP)["res"].mean().sort_values().index
        data = [res.loc[res[GROUP] == g, "res"].values for g in order if (res[GROUP] == g).sum() >= 5]
        ax[k].boxplot(data, showfliers=False); ax[k].axhline(0, color="k", lw=0.6); ax[k].set_title(f"random forest residuals, {title}", fontsize=9)
        ax[k].set_xticks([]); ax[k].set_xlabel("studies with ≥5 mixes, sorted by mean residual")
    ax[0].set_ylabel("measured − predicted (MPa)"); plt.tight_layout(); plt.savefig("figures/F4_residuals_by_study.png", dpi=200); plt.close()
    json.dump({"seed": a.seed, "xgboost": HAVE_XGB, "features": RAW_FEATURES, "categorical": CATEGORICAL,
               "dropped": DROPPED_SPARSE, "between_study_variance_share": round(float(share), 3)}, open("results/run_config.json", "w"), indent=2)
    print("\nwritten: results/, figures/")

if __name__ == "__main__":
    main()
