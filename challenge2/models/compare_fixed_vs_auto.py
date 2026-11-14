"""Matched outer-fold comparison of fixed models vs automatic team selection.

Save alongside team_pipeline_audit.py and compare_team_features.py in
challenge2/models/. Run using Python 3.11 / scikit-learn >= 1.5:

    python challenge2/models/compare_fixed_vs_auto.py --mode quick

Each approach uses IDENTICAL outer validation folds. Model hyperparameters,
thresholds, and automatic model choice are determined using outer-TRAINING
rows only. Never reads challenge2_test.csv or changes purchase_prediction.py.

Quick mode uses the same reduced four-model settings as team_pipeline_audit.py;
it is NOT the team's full eight-grid production workflow.

The data and feature choices have been explored previously, so all
results remain exploratory. See the README note printed at the end.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV)

from compare_team_features import locate_data
from team_pipeline_audit import (search_specs, team_pipe,
                                 choose_team_candidate)

SEED = 42


def get_ensemble(fits, Xtrain):
    """Same three tuned tree members as the team's soft-voting candidate."""
    names = ['Random Forest', 'Gradient Boosting', 'HistGradientBoosting']
    members = [(n.lower().replace(' ', '_'),
                clone(fits[n].best_estimator_.named_steps['model']))
               for n in names]
    return team_pipe('Soft-Voting Ensemble',
                     VotingClassifier(estimators=members, voting='soft', n_jobs=1),
                     Xtrain)


def fit_and_score(name, estimator, Xtrain, ytrain, Xval, yval, inner_cv):
    """Tune threshold on training folds, score only on outer validation rows."""
    model = TunedThresholdClassifierCV(
        clone(estimator), scoring='f1', cv=inner_cv, thresholds=100, n_jobs=1)
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xval)
    probabilities = model.predict_proba(Xval)[:, 1]
    return dict(model=name,
                f1=float(f1_score(yval, predictions, zero_division=0)),
                precision=float(precision_score(yval, predictions, zero_division=0)),
                recall=float(recall_score(yval, predictions, zero_division=0)),
                average_precision=float(average_precision_score(yval, probabilities)),
                threshold=float(model.best_threshold_))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--outer-folds', type=int, default=3)
    args = parser.parse_args()
    if args.outer_folds < 2:
        parser.error('Use at least 2 outer folds')

    csv_path = locate_data()
    df = pd.read_csv(csv_path).dropna(subset=['Revenue'])
    X = df.drop(columns=['Revenue', 'ID'], errors='ignore')
    y = df['Revenue'].astype(bool)

    inner_folds = 2 if args.mode == 'quick' else 5
    model_selection_folds = 2 if args.mode == 'quick' else 5
    threshold_grid = (np.round(np.arange(0.1, 0.901, 0.025), 3)
                      if args.mode == 'quick'
                      else np.round(np.arange(0.05, 0.951, 0.005), 3))
    if y.nunique() != 2 or y.value_counts().min() < (args.outer_folds * 4):
        parser.error('Not enough examples of each class for cross-validation')

    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True,
                              random_state=2027)
    specs = search_specs(args.mode)
    print('=== FIXED MODEL VS AUTOMATIC SELECTION (MATCHED OUTER FOLDS) ===', flush=True)
    print(f'Mode={args.mode} rows={len(df)} outer_folds={args.outer_folds} '
          f'inner_folds={inner_folds}', flush=True)
    print('All models use the team feature recipe; only model-selection changes.', flush=True)
    print('No hidden test data, no submission, and no team source modifications.', flush=True)
    if args.mode == 'full':
        print('WARNING: full mode is VERY expensive and may take a long time.', flush=True)
    rows = []

    for i, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), 1):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xva, yva = X.iloc[valid_idx], y.iloc[valid_idx]
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                   random_state=SEED + i)
        print(f'\nOuter fold {i}/{args.outer_folds}: tuning model configurations...', flush=True)
        start = time.time()
        fits = {}
        for name, (learner, grid) in specs.items():
            search = GridSearchCV(team_pipe(name, learner, Xtr), grid,
                                  scoring='f1', cv=inner_cv, refit=True,
                                  n_jobs=1, error_score='raise')
            search.fit(Xtr, ytr)
            fits[name] = search
            print(f'  {name:23s} training-only tuning F1={search.best_score_:.4f}',
                  flush=True)

        fixed = {name: fit.best_estimator_ for name, fit in fits.items()}
        fixed['Soft-Voting Ensemble'] = get_ensemble(fits, Xtr)

        auto_name, auto_estimator, _, _ = choose_team_candidate(
            fits, Xtr, ytr, inner_cv, threshold_grid, model_selection_folds)
        print(f'  Training-only automatic choice: {auto_name}', flush=True)

        for name, estimator in fixed.items():
            row = fit_and_score(name, estimator, Xtr, ytr, Xva, yva, inner_cv)
            row.update(fold=i, selected_model=auto_name, approach='Fixed')
            rows.append(row)
            print(f'  Fixed {name:23s} outer F1={row["f1"]:.4f}', flush=True)

        # The team selector may choose a model also present in fixed, but we
        # evaluate the same estimated procedure under the 'Auto' label for
        # easy fold-by-fold comparison. Avoid redundant fitting when possible.
        if auto_name in fixed:
            source_row = next(r for r in reversed(rows)
                              if r['fold'] == i and r['model'] == auto_name)
            auto_row = source_row.copy()
        else:
            auto_row = fit_and_score(auto_name, auto_estimator,
                                     Xtr, ytr, Xva, yva, inner_cv)
        auto_row.update(model='Automatic Selection', approach='Auto',
                        selected_model=auto_name)
        rows.append(auto_row)
        print(f'  AUTO ({auto_name}): outer F1={auto_row["f1"]:.4f} '
              f'({time.time() - start:.0f}s)', flush=True)

    result = pd.DataFrame(rows)
    metrics = ['f1', 'precision', 'recall', 'average_precision']
    summary = result.groupby('model')[metrics].agg(['mean', 'std'])
    print('\n=== SAME-FOLD OUTER VALIDATION METRICS ===')
    print(summary.round(4).to_string())
    f1_table = result.pivot(index='fold', columns='model', values='f1')
    reference = f1_table['Automatic Selection']
    diff = f1_table.drop(columns='Automatic Selection').subtract(reference, axis=0)
    print('\n=== PAIRED F1 DIFFERENCE: FIXED MODEL MINUS AUTO ===')
    print(diff.round(4).to_string())
    print('\nMean paired difference vs automatic selection:')
    for col in diff.columns:
        print(f'  {col}: {diff[col].mean():+.4f} F1, '
              f'wins {(diff[col] > 0).sum()}/{len(diff)} folds')

    destination = Path(__file__).resolve().parent
    per_fold_path = destination / f'fixed_vs_auto_{args.mode}_results.csv'
    diff_path = destination / f'fixed_vs_auto_{args.mode}_paired.csv'
    result.to_csv(per_fold_path, index=False)
    diff.to_csv(diff_path)
    print(f'\nSaved: {per_fold_path}')
    print(f'Saved: {diff_path}')
    print('CAUTION: these are exploratory comparisons of previously explored',
          'features/data, NOT an independent final estimate or proof of a win.')
    print('Do not modify the team model based on small differences alone.')


if __name__ == '__main__':
    main()
