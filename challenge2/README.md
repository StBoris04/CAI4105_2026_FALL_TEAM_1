# Challenge 2: Online shopping purchase prediction

Predict whether a shopping session results in a purchase (`Revenue`). This
workspace is separate from Challenge 1. Add the instructor-provided files here
when available:

- `data/challenge2_train.csv` (labeled training data)
- `data/challenge2_test.csv` (unlabeled final prediction data)

These CSV files are ignored by Git so the data can be added locally without
committing course-provided data. Keep the final `submissions/predictions.csv`
submission in Git.

## Modeling plan

- Exclude `ID` from predictors and retain it only to identify test predictions.
- Use a stratified validation strategy with F1 for the positive `Revenue=True`
  class as the primary metric. Report precision, recall, additional metrics, and
  the confusion matrix; do not use accuracy alone.
- Compare Logistic Regression, Naïve Bayes, K-Nearest Neighbors, Support Vector
  Machine, Decision Tree, Random Forest, and Gradient Boosting.
- Fit preprocessing inside model pipelines, tune selected models with Grid
  Search, compare train and validation results, then refit the selected model on
  all labeled data.
- Generate one `ID,Revenue` row per test observation.

## Setup and run

Use Python 3.11+ and install the project dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib jupyter
```

The modeling entry point will be `src/train.py`; after adding both CSVs, run
`python src/train.py` from this folder. The workflow will save predictions under
`submissions/` and validation outputs under `artifacts/`.

The code and notebook are intentionally left as a scaffold until the data is
available, so schema-dependent preprocessing and model comparisons can be based
on the actual files rather than guessed column types.
