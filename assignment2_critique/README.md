# Assignment 2 – Paper critique: T-BoostNet reality check

Reproduction of T-BoostNet (Hassan et al., 2026, *Composite Structures* 378, 119906) on the Assignment 1 data and setup.

## Files

| File | What it does |
| --- | --- |
| `tboostnet.py` | Faithful approximation of T-BoostNet (Section 4.6, Eqs. 4-10). Two readings of the unspecified attention input: `literal` (one vector per sample, as Eqs. 6-7 are written) and `tokens` (one token per feature). |
| `reality_check.py` | 3 setups (`paper`, `paper_no_density`, `a1`) x 3 splits (`random`, `by_mix`, `by_study`) x 5 models. |
| `reality_check_draws.py` | The by-study result over 8 re-drawn study-fold assignments (same idea as `ablation_folds.py`). |
| `offset_share.py` | How much of the error on an unseen study is one shift per study: one-way random-effects share of the held-out-study residuals, over the same 8 draws (A1 XGBoost: 63%; the paper's XGBoost: 61%). |
| `results/reality_check.csv` | Pooled out-of-fold R2 / MAE / RMSE, plus fold mean and sd, seed 42. |
| `results/reality_check_folds.csv` | One row per fold. |
| `results/reality_check_draws.csv` | One row per draw, setup and model. |
| `results/offset_share.csv` | One row per draw and model: by-study R2, offset share, random-effects share and the offset-known ceiling. |

## Reproduce

Assignment 2 sits in its own folder, `assignment2_critique/`, and uses `run_jiang.py` from the repository root. Run the scripts from inside this folder, with the data at `data/pone.0310422.s002.xlsx` in the repository root; results are written to `assignment2_critique/results/`.

```
cd assignment2_critique
pip install torch            # CPU is enough; tested with torch 2.15.0
python reality_check.py --xlsx ../data/pone.0310422.s002.xlsx --seed 42          # about 10 min on 2 cores
python reality_check_draws.py --xlsx ../data/pone.0310422.s002.xlsx --draws 8    # about 15 min on 2 cores
python offset_share.py --xlsx ../data/pone.0310422.s002.xlsx                     # about 2 min
```

Tested with torch 2.15.0, scikit-learn 1.9.0, xgboost 3.4.1, pandas 3.0.3, numpy 2.4.6 (Python 3.13 on Linux), and rerun by the author on Windows with Python 3.14. Results files are appended to, so delete them before a fresh run.

## Choices the paper does not report (fixed before running, not tuned)

- Projection size p = 64 (the paper's own Transformer baseline, Table 3); FFN 64-32 with ReLU; Adam; batch 64; up to 400 epochs; early stopping with patience 30 on an inner 10 % validation split, drawn the same way as the outer split (random, by mix or by study); no dropout.
- Leaf embedding initialised like a linear layer on the one-hot leaves (Eq. 5).
- The XGBoost stage is fitted first and frozen: a tree ensemble cannot be trained jointly with a network by gradient descent.
- A small probe of patience, batch size and validation fraction moved the random-split R2 between 0.824 and 0.838; the pre-set defaults were kept rather than picking the best on the evaluation folds.

## Setups

- `paper`: the 17 inputs of the paper's SHAP plot (Fig. 13), converted (cube-equivalent) strength as target, log(1+y) target, median imputation and robust scaling fitted inside each fold.
- `paper_no_density`: the same without density.
- `a1`: the Assignment 1 inputs and preprocessing (`run_jiang.pipeline`), raw strength.

Mix identity (for `by_mix`) = same study, same proportions, chemistry, curing, specimen type and size; only the test age differs. 1,122 rows hold 610 distinct mixes.

## AI use

Code written with Claude (Anthropic) and reviewed by the author; every reported number comes from the files in `results/`. The code was run by the author to validate the results.
