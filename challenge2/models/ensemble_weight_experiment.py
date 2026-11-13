"""Challenge 2: nested comparison of equal versus training-selected ensemble weights.

Put this alongside compare_team_features.py and team_pipeline_audit.py inside
challenge2/models/; run from anywhere:

    python challenge2/models/ensemble_weight_experiment.py --mode quick

The team feature recipe and three candidate learners remain unchanged. Only the
soft-voting weights are compared. Each outer validation fold is held aside while
hyperparameters, ensemble weights, and the decision threshold are selected.

The data has already been explored in past experiments, so treat results as
exploratory; this does not replicate the team's complete eight-model workflow.
Does not access challenge2_test.csv, predictions.csv, or purchase_prediction.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from compare_team_features import locate_data
from team_pipeline_audit import search_specs, team_pipe

TREE_MODELS = ('Random Forest', 'Gradient Boosting', 'HistGradientBoosting')
# Small, predeclared grid. Avoid searching arbitrary weights on validation rows.
WEIGHTS = (
    (1, 1, 1),  # equal weights — the team's current recipe
    (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (2, 2, 1), (2, 1, 2), (1, 2, 2),
)
THRESHOLDS = np.arange(0.20, 0.651, 0.025).round(3)


def combine(probabilities, weights):
    """Weighted mean of three class-True probability vectors."""
    return np.average(np.column_stack(probabilities), axis=1, weights=weights)


def choose_threshold(y, probabilities):
    """Select a cutoff on OUTER-TRAINING out-of-fold predictions only."""
    values = [f1_score(y, probabilities >= t, zero_division=0) for t in THRESHOLDS]
    idx = int(np.argmax(values))
    return float(THRESHOLDS[idx]), float(values[idx])


def score(y, probabilities, threshold):
    pred = probabilities >= threshold
    return dict(f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                average_precision=float(average_precision_score(y, probabilities)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--outer-folds', type=int, default=3)
    args = parser.parse_args()
    if args.outer_folds < 2:
        parser.error('Use at least 2 outer folds.')

    path = locate_data()
    train = pd.read_csv(path).dropna(subset=['Revenue'])
    X = train.drop(columns=['Revenue', 'ID'], errors='ignore')
    y = train['Revenue'].astype(bool)
    inner_folds = 2 if args.mode == 'quick' else 5
    if y.nunique() != 2 or y.value_counts().min() < 4 * max(inner_folds, args.outer_folds):
        parser.error('Insufficient examples per class for this experiment.')

    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True,
                            random_state=2027)
    grids = search_specs(args.mode)
    print('=== EQUAL VS TUNED SOFT-VOTING WEIGHTS ===', flush=True)
    print(f'Mode: {args.mode}; rows: {len(train)}; outer folds: {args.outer_folds}; '
          f'inner folds: {inner_folds}', flush=True)
    print('Models:', ', '.join(TREE_MODELS), flush=True)
    print('Features, learners, and outer-fold rows held constant.', flush=True)
    print('Thresholds and weights chosen ONLY using outer-training predictions.', flush=True)
    print('Hidden test file is never opened. This is exploratory, NOT proof of a win.', flush=True)

    rows = []
    for fold, (train_indices, valid_indices) in enumerate(outer.split(X, y), 1):
        Xtr, ytr = X.iloc[train_indices], y.iloc[train_indices]
        Xva, yva = X.iloc[valid_indices], y.iloc[valid_indices]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                random_state=42 + fold)
        print(f'\nFold {fold}/{args.outer_folds}: tune three learners...', flush=True)
        prob_train = []
        prob_valid = []
        for name in TREE_MODELS:
            learner, grid = grids[name]
            search = GridSearchCV(team_pipe(name, learner, Xtr), grid, scoring='f1',
                                  cv=inner, n_jobs=1, refit=True, error_score='raise')
            search.fit(Xtr, ytr)
            # OOF predictions are produced with an estimator fitted only to the
            # training section of each inner fold. Settings were chosen using
            # the full outer-training section; outer validation is still held out.
            train_p = cross_val_predict(clone(search.best_estimator_), Xtr, ytr,
                                         cv=inner, method='predict_proba',
                                         n_jobs=1)[:, 1]
            valid_p = search.best_estimator_.predict_proba(Xva)[:, 1]
            prob_train.append(train_p)
            prob_valid.append(valid_p)
            print(f'  {name}: inner-tuning F1={search.best_score_:.4f}', flush=True)

        candidate_scores = []
        for weights in WEIGHTS:
            p = combine(prob_train, weights)
            threshold, cv_f1 = choose_threshold(ytr, p)
            candidate_scores.append((cv_f1, weights, threshold))
        # Fixed order resolves ties consistently and favors the equal recipe.
        equal = next(row for row in candidate_scores if row[1] == (1, 1, 1))
        selected = max(candidate_scores, key=lambda row: row[0])
        for label, (train_f1, weights, threshold) in (
            ('Equal weights', equal), ('Selected weights', selected)
        ):
            p = combine(prob_valid, weights)
            metrics = score(yva, p, threshold)
            rows.append(dict(fold=fold, approach=label,
                             weights=':'.join(map(str, weights)),
                             threshold=threshold,
                             outer_train_oof_f1=train_f1, **metrics))
            print(f'  {label:17s} weights={weights} cutoff={threshold:.3f} '
                  f'outer F1={metrics["f1"]:.4f}, '
                  f'P={metrics["precision"]:.4f}, R={metrics["recall"]:.4f}',
                  flush=True)

    results = pd.DataFrame(rows)
    print('\n=== OUTER-FOLD MEANS (MATCHED FOLDS) ===')
    print(results.groupby('approach')[['f1', 'precision', 'recall', 'average_precision']]
          .agg(['mean', 'std']).round(4).to_string())
    by_fold = results.pivot(index='fold', columns='approach', values='f1')
    by_fold['selected_minus_equal_f1'] = by_fold['Selected weights'] - by_fold['Equal weights']
    print('\n=== PAIRED F1 DIFFERENCE (SELECTED MINUS EQUAL) ===')
    print(by_fold.round(4).to_string())
    diff = by_fold['selected_minus_equal_f1']
    print(f'Mean delta={diff.mean():+.4f}; selected wins {(diff > 0).sum()}/'
          f'{len(diff)}, ties={(diff == 0).sum()}.')
    print('A tiny difference on these already-explored data is NOT a reliable win.')
    folder = Path(__file__).resolve().parent
    result_file = folder / f'ensemble_weights_{args.mode}_results.csv'
    paired_file = folder / f'ensemble_weights_{args.mode}_paired.csv'
    results.to_csv(result_file, index=False)
    by_fold.to_csv(paired_file)
    print('Saved:', result_file)
    print('Saved:', paired_file)
    print('No team source or submission files were modified.')


if __name__ == '__main__':
    main()
