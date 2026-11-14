# Challenge 2 — Full-Grid Model Validation Report

**Project:** CAI4105 Fall 2026, Team 1  
**Experiment:** Nested outer-fold audit of the purchase (`Revenue=True`) classifier  
**Source:** `challenge2/models/team_pipeline_audit.py`  
**Status:** Three of three full-mode outer folds completed; exploratory results

## Executive summary

The separate audit completed **three outer validation folds** on the supplied training dataset (9,247 rows; positive-class prevalence approximately 15.5%). Within each outer training split, the audit tuned the original team's eight algorithm families using five-fold inner cross-validation, then compared five final candidate model recipes and tuned the decision threshold without using the outer validation targets.

The **mean outer-fold F1 for `Revenue=True` was 0.6924**, with mean precision **0.6394**, mean recall **0.7561**, and mean average precision **0.7514**. Random Forest was selected in two folds and Gradient Boosting in one. These numbers summarize a separate audit implementation; they do **not** establish a higher-scoring replacement for the team's submission model.

## Full-mode outer-fold results

| Outer fold | Selected model | F1 | Precision | Recall | Average precision | Decision threshold | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Random Forest | 0.7077 | 0.6630 | 0.7589 | 0.7879 | 0.5172 | 764 s |
| 2 | Random Forest | 0.6837 | 0.6433 | 0.7296 | 0.7323 | 0.5242 | 4,293 s |
| 3 | Gradient Boosting | 0.6857 | 0.6118 | 0.7799 | 0.7341 | 0.2818 | 711 s |
| **Mean** | — | **0.6924** | **0.6394** | **0.7561** | **0.7514** | — | — |

*Means are unweighted arithmetic averages of the reported fold-level scores. Values are rounded. The reported average precision is a separate ranking metric and is not the same as the precision column.*

## Matched quick-mode reference

| Outer fold | Quick-mode F1 | Full-mode F1 | Full minus quick |
|---|---:|---:|---:|
| 1 | 0.7008 | 0.7077 | +0.0069 |
| 2 | 0.6805 | 0.6837 | +0.0032 |
| 3 | 0.6746 | 0.6857 | +0.0111 |
| **Mean** | **0.6853** | **0.6924** | **+0.0071** |

Full mode scored higher on these three matched splits, but the observed differences are small and this is not independent proof that the larger search improves future performance.

## Evaluation design and reproducibility

- **Target:** `Revenue=True`; **training rows:** 9,247; approximate positive share: 15.5%.
- **Outer validation:** three folds; one outer fold run per invocation and saved as a separate result CSV.
- **Inner tuning:** five-fold cross-validation in full mode.
- **Algorithms tuned:** Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting.
- **Final candidate comparison:** five candidate recipes, including a Soft-Voting Ensemble; candidate and threshold selection occur within outer-training rows.
- **Evaluation:** Each selected model is scored on its outer validation fold only after training-only tuning and selection.
- **Data isolation:** The hidden `challenge2_test.csv` was not read by the audit. The audit did not produce a submission or execute or modify `purchase_prediction.py`.
- **Runtime settings:** Fold 1 and Fold 2 ran before the grid-job option was added; Fold 3 ran with `--grid-jobs 2`. These are not controlled runtime comparisons.

### Reproduce each fold

From the root of the cloned repository, using an environment with the required Python packages:

```powershell
python .\challenge2\models\team_pipeline_audit.py --mode full --outer-folds 3 --only-fold 1
python .\challenge2\models\team_pipeline_audit.py --mode full --outer-folds 3 --only-fold 2
python .\challenge2\models\team_pipeline_audit.py --mode full --outer-folds 3 --only-fold 3 --grid-jobs 2
```

Saved fold files:

- `challenge2/models/team_pipeline_audit_full_fold1_results.csv`
- `challenge2/models/team_pipeline_audit_full_fold2_results.csv`
- `challenge2/models/team_pipeline_audit_full_fold3_results.csv`

## Interpretation and limitations

1. **The selected model varies by fold.** Random Forest was selected twice and Gradient Boosting once; this does not prove a single algorithm is always best.
2. **These are exploratory estimates.** The dataset and modeling options have been examined across many experiments. Only three outer folds were used, so modest changes in mean F1 should not be treated as conclusive.
3. **The audit is a separate implementation.** It uses the team's original full hyperparameter grids but is not an exact end-to-end execution of `purchase_prediction.py`.
4. **No claim about hidden-test performance.** Hidden-test labels are unavailable, and no hidden-test evaluation was performed.
5. **Full-mode runtime varied substantially.** Fold 2 took much longer than Fold 1 or Fold 3; parallelization alone cannot explain or validate the difference without controlled profiling.

## Recommended team decision

**Keep the existing submission model unchanged based on this report alone.** Use these results to document the evaluation procedure and performance variation. Before replacing model-selection logic, test the proposed change using the same prespecified outer folds and selection protocol; prefer a simpler approach when performance differences are small. Separately ensure the submission CSV is saved to the documented path and contains the required `ID` and `Revenue` columns.

## GitHub history

The completed full-fold results were committed on branch `feature/model-validation-experiments`, including commit `cd10419` (full-grid evaluation completion and timing diagnostics). This report is intended for review in the same branch before merging changes to `main`.
