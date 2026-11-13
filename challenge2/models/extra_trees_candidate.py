"""Challenge 2: evaluate Extra Trees as an additional model candidate.

Place beside compare_team_features.py in challenge2/models/.
From the Git repository root run:

    python challenge2/models/extra_trees_candidate.py --mode quick

This compares three FIXED learner recipes on identical outer validation folds:
Random Forest, Extra Trees, and a soft-voting ensemble of the team's three
established tree-based models. For each candidate, the decision threshold is
picked using out-of-fold predictions on the OUTER TRAINING rows only. Also
reports a training-only selection of the candidate with the best inner OOF F1.

Important limitations:
- The settings are a practical experiment, NOT a reproduction of the team's
  full hyperparameter grids or full model-selection workflow.
- The same labeled dataset has already been used for exploratory work. Results
  are not independent final proof of an improvement.
- Never reads challenge2_test.csv, modifies purchase_prediction.py, or saves
  submission predictions. Requires pandas/scikit-learn and existing helpers.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier,
)
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from compare_team_features import locate_data, pipeline

SEED = 42
CANDIDATES = ('Random Forest', 'Extra Trees', 'Soft-Voting Ensemble')


def make_candidate(name, X, trees):
    """Use the team's exact feature recipe for every learner."""
    forest = RandomForestClassifier(
        n_estimators=trees, max_features='sqrt', min_samples_leaf=3,
        random_state=SEED, n_jobs=1,
    )
    if name == 'Random Forest':
        model = forest
    elif name == 'Extra Trees':
        model = ExtraTreesClassifier(
            n_estimators=trees, max_features='sqrt', min_samples_leaf=3,
            random_state=SEED, n_jobs=1,
        )
    elif name == 'Soft-Voting Ensemble':
        model = VotingClassifier(estimators=[
            ('rf', forest),
            ('gb', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=2,
                subsample=0.8, random_state=SEED,
            )),
            ('hgb', HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.1, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=SEED,
            )),
        ], voting='soft', n_jobs=1)
    else:
        raise ValueError(f'Unknown candidate: {name}')
    return pipeline('Random Forest', 'Team features', X).set_params(model=model)


def metrics(y_true, probabilities, threshold):
    predictions = probabilities >= threshold
    return {
        'f1': float(f1_score(y_true, predictions, zero_division=0)),
        'precision': float(precision_score(y_true, predictions, zero_division=0)),
        'recall': float(recall_score(y_true, predictions, zero_division=0)),
        'average_precision': float(average_precision_score(y_true, probabilities)),
    }


def run(mode='quick', outer_folds=None):
    data_path = locate_data()
    frame = pd.read_csv(data_path)
    if not {'Revenue', 'ID'}.issubset(frame.columns):
        raise ValueError('Training data must include ID and Revenue columns')
    if frame['Revenue'].isna().any():
        raise ValueError('Training Revenue has missing values')
    # Enforce explicit binary target encoding; bool("False") would be True.
    allowed = {True, False, 0, 1}
    if not set(frame['Revenue'].unique()).issubset(allowed):
        raise ValueError('Revenue must be parsed as Boolean or numeric 0/1')

    X = frame.drop(columns=['ID', 'Revenue'])
    y = frame['Revenue'].astype(bool)
    n_outer = outer_folds or (3 if mode == 'quick' else 5)
    n_inner = 2 if mode == 'quick' else 3
    trees = 100 if mode == 'quick' else 250
    if n_outer < 2:
        raise ValueError('outer-folds must be >= 2')
    if y.nunique() != 2 or y.value_counts().min() < n_outer * n_inner * 2:
        raise ValueError('Not enough examples of both classes for nested CV')
    thresholds = np.round(np.arange(0.20, 0.651, 0.025), 3)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=2027)

    print('=== EXTRA TREES CANDIDATE: NESTED THRESHOLD VALIDATION ===', flush=True)
    print(f'Mode={mode}; rows={len(y)}; outer folds={n_outer}; '
          f'inner folds={n_inner}; trees={trees}', flush=True)
    print('Every candidate uses the team feature recipe and identical outer splits.', flush=True)
    print('Thresholds AND the candidate choice use outer-TRAINING rows only.', flush=True)
    print('No hidden test data, no submission, no changes to team source.', flush=True)
    print('This is exploratory, NOT a full reproduction of team hyperparameter tuning.\n', flush=True)

    rows = []
    for fold, (i_train, i_valid) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[i_train], y.iloc[i_train]
        X_valid, y_valid = X.iloc[i_valid], y.iloc[i_valid]
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                   random_state=SEED + fold)
        fold_rows = {}
        for name in CANDIDATES:
            estimator = make_candidate(name, X_train, trees)
            # Inner OOF probabilities: preprocessing and fitting happen per split.
            train_oof = cross_val_predict(
                clone(estimator), X_train, y_train, cv=inner_cv,
                method='predict_proba', n_jobs=1,
            )[:, 1]
            inner_scores = np.asarray([
                f1_score(y_train, train_oof >= cut, zero_division=0)
                for cut in thresholds
            ])
            selected_cutoff = float(thresholds[int(np.argmax(inner_scores))])
            trained = clone(estimator).fit(X_train, y_train)
            outer_proba = trained.predict_proba(X_valid)[:, 1]
            outer_metrics = metrics(y_valid, outer_proba, selected_cutoff)
            row = dict(
                fold=fold, candidate=name, threshold_from_train=selected_cutoff,
                training_oof_f1=float(inner_scores.max()), **outer_metrics,
            )
            rows.append(row)
            fold_rows[name] = row
            print(f'Fold {fold}/{n_outer} | {name:21s} '
                  f'inner F1={row["training_oof_f1"]:.4f} '
                  f'cutoff={selected_cutoff:.3f} '
                  f'OUTER F1={row["f1"]:.4f} '
                  f'P={row["precision"]:.4f} R={row["recall"]:.4f}', flush=True)

        # Training-only model choice, evaluated on the SAME outer fold.
        winner = max(CANDIDATES,
                     key=lambda name: fold_rows[name]['training_oof_f1'])
        selected_row = dict(fold_rows[winner])
        selected_row['candidate'] = 'Training-selected candidate'
        selected_row['selected_name'] = winner
        rows.append(selected_row)
        print(f'  Training selected: {winner}; outer F1={selected_row["f1"]:.4f}\n', flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby('candidate', sort=False)[
        ['f1', 'precision', 'recall', 'average_precision']
    ].agg(['mean', 'std'])
    print('\n=== SAME-FOLD VALIDATION RESULTS ===')
    print(summary.round(4).to_string())
    pivot = results.pivot(index='fold', columns='candidate', values='f1')
    delta = pivot['Extra Trees'] - pivot['Soft-Voting Ensemble']
    print('\n=== EXTRA TREES MINUS ENSEMBLE: PAIRED F1 ===')
    print(delta.round(4).to_string())
    print(f'Mean difference={delta.mean():+.4f}; '
          f'Extra Trees wins={(delta > 0).sum()}/{len(delta)} folds.')
    print('Small gains are NOT proof; repeated experimentation on the same data '
          'can lead to optimistic conclusions.')

    destination = Path(__file__).resolve().parent
    result_path = destination / f'extra_trees_{mode}_results.csv'
    pair_path = destination / f'extra_trees_{mode}_paired.csv'
    results.to_csv(result_path, index=False)
    pivot.assign(extra_minus_ensemble=delta).to_csv(pair_path)
    print(f'Saved: {result_path}\nSaved: {pair_path}', flush=True)
    return results


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--mode', choices=('quick', 'full'), default='quick')
    arg.add_argument('--outer-folds', type=int, default=None,
                     help='Optional fold count; default 3 quick or 5 full')
    args = arg.parse_args()
    run(mode=args.mode, outer_folds=args.outer_folds)


if __name__ == '__main__':
    main()
