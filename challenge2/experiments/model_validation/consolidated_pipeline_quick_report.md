# Challenge 2 — Consolidated Pipeline Validation

**Run (UTC):** 2026-11-14T00:05:18+00:00
**Mode:** quick | **Outer folds:** 3 | **Inner folds:** 2
**Rows:** 9247 | **Positive rate:** 15.48%
**Training CSV SHA256:** `c93858d38b60a8924f420b52280971d41f120af2d44dc64f5997df3bba0d3d51`
**Environment:** Python 3.11.9 / scikit-learn 1.5.2

## Matched outer-fold results

| Approach | Mean F1 | Mean precision | Mean recall | Mean AP |
|---|---:|---:|---:|---:|
| Automatic Selection | 0.6868 | 0.6496 | 0.7317 | 0.7521 |
| Fixed Soft-Voting Ensemble | 0.6903 | 0.6401 | 0.7519 | 0.7576 |
| Fixed Random Forest | 0.6846 | 0.6491 | 0.7282 | 0.7530 |

## Per-fold scores

| Fold | Automatically selected model | Auto F1 | Ensemble F1 | Random Forest F1 | Auto − ensemble |
|---:|---|---:|---:|---:|---:|
| 1 | Random Forest | 0.6997 | 0.7081 | 0.6997 | -0.0084 |
| 2 | Soft-Voting Ensemble | 0.6787 | 0.6787 | 0.6680 | +0.0000 |
| 3 | Gradient Boosting | 0.6820 | 0.6841 | 0.6862 | -0.0021 |

**Mean paired Auto − ensemble F1:** -0.0035; auto won 0/3 folds; tied 1/3.

## Method and interpretation

- All three approaches were evaluated on exactly the same outer validation rows.
- Hyperparameter tuning, candidate ranking, and threshold tuning used only the outer training rows.
- The ensemble uses tuned Random Forest, Gradient Boosting and HistGradientBoosting members.
- The automatic selection is the team-inspired audit implementation; it does **not** execute the original script.
- Quick mode uses reduced grids; full mode uses the eight original model families and grids, but is still a reimplementation.
- The test CSV is never read, no submission is produced, and the original team source is unchanged.
- This training dataset and the candidate recipes have been inspected repeatedly. Report these as exploratory results, **not proof** of a new improvement or a final unbiased score.
- The original team code performs hyperparameter/model/threshold selection with a different detailed sequence; full equivalence has not been established.

## Suggested action

Keep the production model unchanged unless a meaningful advantage is repeatable and the team reviews it. If the scores are similar, prefer the simpler and more reproducible choice.
