# CSBP 711 – Assignment 1: Compressive strength of fly-ash geopolymer concrete from mix design

**Student:** Hakem Mohammad Alkhraisha (700043202), working alone.  
**Task:** supervised regression (compressive strength in MPa from mix proportions, fly-ash chemistry and curing).

## Dataset
- Jiang P., Zhao D., Jin C., Ye S., Luan C., Tufail R. F. (2024). *Compressive strength prediction and low-carbon optimization of fly ash geopolymer concrete based on big data and ensemble learning.* PLOS ONE 19(9): e0310422. https://doi.org/10.1371/journal.pone.0310422
- File used: the paper's **S1 Table**, the full dataset (file `pone.0310422.s002.xlsx`; the file numbering is offset by the S1 Appendix), downloaded from the article page on 29 September 2026. The authors' own training/test/validation subsets (files s003–s005) are not used.
- Licence: CC BY 4.0 (PLOS ONE, including supporting information).
- Collection, as described by the authors (Section 2.3): literature retrieved from Web of Science and Google Scholar; only studies reporting every key variable were kept; 1,136 mixes transcribed into one spreadsheet. The sheet has 83 reference blocks, each citing its source paper on its first row (82 distinct papers: one appears under two spellings; one block cites a thesis not in the paper's reference list).
- Size: 1,136 rows x 41 columns as published; 1,122 rows after removing exact duplicates; 17 raw mix variables + specimen type used as features.

Download the S1 Table directly from https://doi.org/10.1371/journal.pone.0310422.s002, create a folder `data/` next to the scripts, and save the file there as `data/pone.0310422.s002.xlsx` (not redistributed here).

## How to reproduce
```
pip install -r requirements.txt
python run_jiang.py --xlsx data/pone.0310422.s002.xlsx --seed 42
python ablation_depth.py --xlsx data/pone.0310422.s002.xlsx --seed 42
python ablation_folds.py --xlsx data/pone.0310422.s002.xlsx --draws 8   # needs scikit-learn >= 1.6
python ablation_boosting.py --xlsx data/pone.0310422.s002.xlsx
python audit_overlap.py --xlsx data/pone.0310422.s002.xlsx
```
Outputs: `results/data_audit.txt`, `results/comparison.csv`, `results/ablation.csv`, `results/run_config.json`, `results/ablation_depth.csv`, `results/ablation_folds.csv`, `results/ablation_boosting.csv`, `results/overlap_audit.txt`, `figures/F1..F4.png`.

The random-split R² values reproduce to three decimals across machines (MAE to two; XGBoost builds differ slightly). The by-study numbers depend on how GroupKFold assigns the 83 reference blocks to five folds, which differs between numpy/scikit-learn builds; the headline table is one such draw, and `ablation_folds.py` re-draws the assignment eight times and reports mean, spread and how often each ranking held (`results/ablation_folds.csv`). `requirements-lock.txt` (from `pip list --format=freeze`) lists the exact package versions used for the reported numbers.

Compute used for the reported numbers: Dell XPS 15 (2019), Windows 11, Anaconda Python 3.14, package versions as in `requirements-lock.txt`; `run_jiang.py` about 1 minute and `ablation_depth.py` about 2 minutes; random forest and XGBoost run on two CPU cores (`n_jobs=2`), ridge and the MLP on one. Seeds are fixed (42). The random-split ranking was unchanged over seeds 1, 7, 42 and 123; seeds do not change the by-study folds, which is why `ablation_folds.py` exists.

## What the script does
1. Data audit: forward-fills the two block-sparse columns (the reference and the specimen size are written on the first row of each study block only; the workbook has no merged cells), ignores two stray unnamed columns, drops two columns that are 79 % missing, imputes molarity (22 % missing) and MgO (15 % missing) with the median inside each training fold, removes 14 exact duplicates, keeps cube/cylinder as a feature (683 cubes, 439 cylinders) rather than applying the authors' fixed cube-conversion factors (x1.18 below 50 MPa, x1.04 above), and reports the share of strength variance that lies between source studies (61 %). Target: strength 1.1–87.4 MPa, mean 36.5 (regression, so no class balance). Leakage hazard: 83 source studies, the largest supplying 113 rows (10 %).
2. Four models with one shared preprocessing pipeline and one seed: ridge regression (baseline), random forest, XGBoost, small MLP.
3. Two evaluations: random 5-fold split (leaky: rows from the same study on both sides) and 5-fold split by source study (honest: whole studies held out).
4. Ablations: (a) study identity added as a feature; (b) a "study-mean" predictor that uses no mix variables at all; (c) `ablation_depth.py`: the random forest limited to depth 8 and 4 under both splits (by-study R2 0.05 -> 0.05 -> 0.02, but the depth-4 forest also falls to 0.58 on the random split, so this test cannot separate depth from capacity and leaves the forest-versus-XGBoost gap unexplained); (d) `ablation_folds.py`: eight re-drawn by-study fold assignments, reporting mean and spread per model; (e) `ablation_boosting.py`: boosted versus bagged trees from the same library at equal depth and sampling, under both splits, testing whether residual fitting explains XGBoost's margin over the forest; `audit_overlap.py` computes the fingerprint-overlap and molarity-derivability numbers quoted on Slide 2.

## Main result (seed 42, one fold draw, run on the student's laptop; by-study scores move with the fold draw — XGBoost first and MLP last hold in every draw, forest versus ridge does not; see ablation_folds.csv)
| Model | R² random split | R² by-study split |
|---|---|---|
| Ridge (baseline) | 0.47 | 0.05 |
| Random forest | 0.80 | 0.05 |
| MLP | 0.80 | -0.86 |
| XGBoost | 0.85 | 0.16 |

Explanation tested, two data properties:
1. Strength in this compilation varies mostly between laboratories (61 % of variance), and the fly-ash and silicate columns barely vary within a study, so on a random split a tree identifies the source. Prediction (a): holding out whole blocks should bring every model to the baseline's level; (b): giving every model the block identity should close the gap. (b) held (ridge 0.47 -> 0.70, trees +-0.01; block mean alone 0.53). (a) held for the MLP, is within fold-draw noise for the forest (0.13 +- 0.06 vs ridge 0.05 +- 0.10 over eight draws), and failed for XGBoost (0.29 +- 0.06, first in 8 of 8 draws). So this property explains the trees' lead over ridge, not XGBoost's margin over the forest.
2. The target is a large between-laboratory offset plus a small within-laboratory signal. Prediction: if fitting residuals is what preserves XGBoost's margin, boosted trees should beat bagged trees from the same library at equal depth (4) and equal sampling when blocks are held out, which removes depth and capacity from the comparison. Result (`ablation_boosting.py`): boosted 0.16 vs bagged -0.03 by block (0.85 vs 0.57 random). Independently averaged trees re-fit the offset in every tree; boosting absorbs it early and then fits the residual, the only part that transfers to a new source. The depth ablation alone could not show this (the depth-4 forest underfits the random split too).

## Limitations (stated honestly)
- The by-study split holds out *reference strings*, not laboratories. Several laboratories contribute more than one paper (one paper also appears under two spellings), and 15 % of held-out rows share their fly-ash oxide fingerprint with a training row (39 % in the worst fold; `results/overlap_audit.txt`). The split removes most, not all, of the laboratory leakage, so the by-study scores are an upper bound on cross-laboratory performance.
- Molarity is median-imputed on 247 rows although it could be derived from the NaOH-solids and water columns for 239 of them; the activator's Na2O/SiO2/H2O columns, which are complete, were dropped as derived. Using them is a possible improvement, not done here.
- Fit times are single runs and vary between runs by tens of percent; model size is that of the final fold's model.
- Hyperparameters were fixed before any result was computed, but no timestamped record exists to prove it.

## Contributions
All data preparation, modelling, analysis and slides: Hakem Mohammad Alkhraisha.

## Use of AI assistants
An AI assistant (Claude, Anthropic) was used to draft the analysis script, to check the dataset structure, and to help structure the explanation and slides. All code was run, checked and interpreted by the student; the dataset choice, the experimental design and the conclusions are the student's own.

## Software cited
Pedregosa et al. (2011) Scikit-learn: Machine Learning in Python, JMLR 12, 2825-2830. Chen & Guestrin (2016) XGBoost: A Scalable Tree Boosting System, KDD '16. McKinney (2010) Data Structures for Statistical Computing in Python (pandas). Harris et al. (2020) Array programming with NumPy, Nature 585, 357-362. Hunter (2007) Matplotlib, CiSE 9, 90-95.

## Licence
Code in this repository: MIT (see `LICENSE`). Data: CC BY 4.0, Jiang et al. 2024, cited above.
