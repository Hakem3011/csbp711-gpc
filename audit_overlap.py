"""
Audit of the by-study split's residual leakage and of the imputation decision. Produces the numbers quoted on Slide 2:
  - share of held-out rows whose fly-ash oxide fingerprint also occurs in the training fold (overall and worst fold),
  - how many rows with missing molarity carry NaOH-solids and water columns from which it could be derived.

Usage:  python audit_overlap.py --xlsx data/pone.0310422.s002.xlsx
Writes: results/overlap_audit.txt
"""
import argparse, os, re, warnings, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import run_jiang as rj
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", required=True); a = ap.parse_args()
os.makedirs("results", exist_ok=True); lines = []
df = rj.load(a.xlsx, [])
X = df[rj.RAW_FEATURES + rj.CATEGORICAL]; y = df[rj.TARGET].values
fp = df[["SiO2 in fly ash (%)", "Al2O3 in fly ash (%)", "CaO in fly ash (%)", "Fe2O3 in fly ash (%)"]].round(2).astype(str).agg("|".join, axis=1)
lines.append(f"distinct fly-ash oxide fingerprints: {fp.nunique()}; used by more than one reference block: {int((df.groupby(fp)[rj.GROUP].nunique() > 1).sum())}")
leak, per = 0, []
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=df[rj.GROUP])):
    seen = set(fp.iloc[tr]); n = int(sum(f in seen for f in fp.iloc[te])); leak += n; per.append(f"fold {k}: {n}/{len(te)} ({n/len(te):.0%})")
lines.append(f"by-study split (GroupKFold on reference block): held-out rows whose fingerprint also occurs in training = {leak}/{len(y)} ({leak/len(y):.0%}); " + "; ".join(per))
raw = pd.read_excel(a.xlsx); raw.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns]
miss = raw["NaOH solution concentration (mol/L)"].isna()
have = miss & raw["NaOH solids (kg/m3)"].notna() & raw["H2O in NaOH solution (kg/m3)"].notna()
lines.append(f"rows with molarity missing: {int(miss.sum())}; of these, rows carrying NaOH solids and water (molarity derivable): {int(have.sum())}; "
             f"activator Na2O/SiO2/H2O columns complete on {int(raw['Na2O in alkaline solution (kg/m3)'].notna().sum())} of {len(raw)} rows")
open("results/overlap_audit.txt", "w").write("\n".join(lines)); print("\n".join(lines))
