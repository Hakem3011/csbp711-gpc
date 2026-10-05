# CSBP 711 – Assignment 1: Compressive strength of fly-ash geopolymer concrete from mix design

**Student:** Hakem Mohammad Alkhraisha (700043202), working alone.  
**Task:** supervised regression (compressive strength in MPa from mix proportions, fly-ash chemistry and curing).

## Dataset
- Jiang P., Zhao D., Jin C., Ye S., Luan C., Tufail R. F. (2024). *Compressive strength prediction and low-carbon optimization of fly ash geopolymer concrete based on big data and ensemble learning.* PLOS ONE 19(9): e0310422. https://doi.org/10.1371/journal.pone.0310422
- File used: the paper's **S1 Table**, the full dataset (file `pone.0310422.s002.xlsx`; the file numbering is offset by the S1 Appendix), downloaded from the article page on 29 September 2026. The authors' own training/test/validation subsets (files s003–s005) are not used.
- Licence: CC BY 4.0 (PLOS ONE, including supporting information).
- Collection, as described by the authors (Section 2.3): literature retrieved from Web of Science and Google Scholar; only studies reporting every key variable were kept; 1,136 mixes from 83 papers transcribed into one spreadsheet; each row cites its source paper.
- Size: 1,136 rows x 41 columns as published; 1,122 rows after removing exact duplicates; 17 raw mix variables + specimen type used as features.

Download the S1 Table directly from https://doi.org/10.1371/journal.pone.0310422.s002, create a folder `data/` next to the scripts, and save the file there as `data/pone.0310422.s002.xlsx` (not redistributed here).

## How to reproduce
```
pip install -r requirements.txt
python run_jiang.py --xlsx data/pone.0310422.s002.xlsx --seed 42
python ablation_depth.py --xlsx data/pone.0310422.s002.xlsx --seed 42
python ablation_folds.py --xlsx data/pone.0310422.s002.xlsx --draws 8   # optional, needs scikit-learn >= 1.6
```
Outputs: `results/data_audit.txt`, `results/comparison.csv`, `results/ablation.csv`, `results/run_config.json`, `results/ablation_depth.csv`, `results/ablation_folds.csv` (optional), `figures/F1..F4.png`.

The random-split numbers reproduce to three decimals across machines. The by-study numbers depend on how GroupKFold assigns the 83 reference blocks to five folds, which can differ between numpy/scikit-learn versions; `ablation_folds.py` re-draws that assignment eight times and reports the spread. `requirements-lock.txt` lists the exact package versions used for the reported numbers.

Compute used for the reported numbers: Dell XPS 15 (2019), Windows 11, Anaconda Python 3.14, package versions as in `requirements-lock.txt`; `run_jiang.py` about 1 minute and `ablation_depth.py` about 2 minutes; random forest and XGBoost run on two CPU cores (`n_jobs=2`), ridge and the MLP on one. Seeds are fixed (42). The random-split ranking was unchanged over seeds 1, 7, 42 and 123; seeds do not change the by-study folds, which is why `ablation_folds.py` exists.

## What the script does
1. Data audit: forward-fills the two block-sparse columns (the reference and the specimen size are written on the first row of each study block only; the workbook has no merged cells), ignores two stray unnamed columns, drops two columns that are 79 % missing, imputes molarity (22 % missing) and MgO (15 % missing) with the median inside each training fold, removes 14 exact duplicates, keeps cube/cylinder as a feature (683 cubes, 439 cylinders) rather than applying the authors' fixed cube-conversion factors (x1.18 below 50 MPa, x1.04 above), and reports the share of strength variance that lies between source studies (61 %). Target: strength 1.1–87.4 MPa, mean 36.5 (regression, so no class balance). Leakage hazard: 83 source studies, the largest supplying 113 rows (10 %).
2. Four models with one shared preprocessing pipeline and one seed: ridge regression (baseline), random forest, XGBoost, small MLP.
3. Two evaluations: random 5-fold split (leaky: rows from the same study on both sides) and 5-fold split by source study (honest: whole studies held out).
4. Ablations: (a) study identity added as a feature; (b) a "study-mean" predictor that uses no mix variables at all; (c) `ablation_depth.py`: the random forest limited to depth 8 and 4 under both splits (by-study R2 0.05 -> 0.05 -> 0.02, but the depth-4 forest also falls to 0.58 on the random split, so this test cannot separate depth from capacity and leaves the forest-versus-XGBoost gap unexplained); (d) `ablation_folds.py`: eight re-drawn by-study fold assignments, reporting mean and spread per model.

## Main result (seed 42, run on the student's laptop; the by-study scores depend on how the 83 studies fall into folds and vary between scikit-learn versions, the ranking does not)
| Model | R² random split | R² by-study split |
|---|---|---|
| Ridge (baseline) | 0.47 | 0.05 |
| Random forest | 0.80 | 0.05 |
| MLP | 0.80 | -0.86 |
| XGBoost | 0.85 | 0.16 |

Explanation tested: strength in this compilation varies mostly between laboratories (61 % of variance); on a random split the tree models recognise the laboratory from near-constant columns (fly-ash chemistry, silicate composition). Prediction: holding out whole studies should bring all models down to the baseline's level, and giving the baseline the study identity should close the gap. Result: the forest and the MLP collapsed (0.05, -0.86); ridge rose from 0.47 to 0.70 with the study identity while the trees moved by 0.01; a predictor using only the study mean scores 0.53. XGBoost kept 0.16, a margin that is consistent across fold draws but is not explained by the laboratory property; the depth ablation could not separate depth from capacity, so the mechanism behind that margin is an open question, not a finding.

## Limitations (stated honestly)
- The by-study split holds out *reference strings*, not laboratories. Several laboratories contribute more than one paper (one paper also appears under two spellings), and 19 % of held-out rows share their fly-ash oxide fingerprint with a training row (52 % in the fold that holds the largest study). The split removes most, not all, of the laboratory leakage, so the by-study scores are an upper bound on cross-laboratory performance.
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
