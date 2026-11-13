"""Challenge 2: evaluate decision-threshold stability using training-only CV.

Save in challenge2/models/ beside compare_team_features.py and
team_pipeline_audit.py. From the cloned repository's root, run:

    python challenge2/models/threshold_stability_check.py --mode quick

This experiment isolates THRESHOLD choice: both predictions on each outer
validation fold come from the EXACT SAME fitted model. It compares the default
0.50 cutoff with one chosen from out-of-fold predictions on outer-training
rows. The test CSV and the team's purchase_prediction.py are never accessed.

The ensemble uses a small, FIXED quick-mode configuration, not the team's
fully tuned model-selection workflow. The data has been explored repeatedly,
so results should be treated as exploratory, not independent proof of a gain.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from compare_team_features import locate_data
from team_pipeline_audit import team_pipe

SEED = 42


def make_estimator(X, model_name):
    """Keep learner and team feature recipe identical across cutoff choices."""
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                random_state=SEED, n_jobs=1)
    if model_name == 'forest':
        learner = rf
    else:
        learner = VotingClassifier(estimators=[
            ('forest', rf),
            ('gradient', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=2,
                subsample=0.8, random_state=SEED)),
            ('hist_gradient', HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.1, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=SEED)),
        ], voting='soft', n_jobs=1)
    return team_pipe('fixed_threshold_check', learner, X)


def metrics(y, prob, cutoff):
    pred = prob >= cutoff
    return {
        'f1': float(f1_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y, prob)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--model', choices=('ensemble', 'forest'), default='ensemble')
    args = parser.parse_args()

    path = locate_data()
    data = pd.read_csv(path)
    if 'Revenue' not in data or 'ID' not in data:
        parser.error('Expected Revenue and ID columns in training CSV.')
    if data['Revenue'].isna().any():
        parser.error('Training target contains missing values.')
    X = data.drop(columns=['Revenue', 'ID'])
    y = data['Revenue'].astype(bool)
    n_outer = 3 if args.mode == 'quick' else 5
    n_inner = 2 if args.mode == 'quick' else 5
    if y.nunique() != 2 or y.value_counts().min() < n_outer * n_inner * 2:
        parser.error('Not enough examples of both classes for selected folds.')
    thresholds = np.round(np.arange(0.20, 0.651, 0.025), 3) if args.mode == 'quick' \
        else np.round(np.arange(0.15, 0.701, 0.01), 3)

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=2027)
    candidate = make_estimator(X, args.model)
    rows = []
    print('=== DEFAULT VS TRAINING-TUNED THRESHOLD ===', flush=True)
    print(f'Model={args.model} mode={args.mode} rows={len(y)} '
          f'outer folds={n_outer} inner folds={n_inner}', flush=True)
    print('Feature processing and model hyperparameters are fixed across both cutoffs.', flush=True)
    print('Only the decision threshold changes. It is tuned on training OOF rows.', flush=True)
    print('The hidden test file and purchase_prediction.py are not read or changed.', flush=True)
    print('This is exploratory, not an independent final model performance estimate.', flush=True)

    for fold, (idx_tr, idx_va) in enumerate(outer_cv.split(X, y), 1):
        Xtr, ytr = X.iloc[idx_tr], y.iloc[idx_tr]
        Xva, yva = X.iloc[idx_va], y.iloc[idx_va]
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                   random_state=SEED + fold)
        # Pick cutoff using only training rows. Never optimize on outer validation.
        p_train_oof = cross_val_predict(
            clone(candidate), Xtr, ytr, cv=inner_cv,
            method='predict_proba', n_jobs=1)[:, 1]
        train_f1 = np.asarray([
            f1_score(ytr, p_train_oof >= cutoff, zero_division=0)
            for cutoff in thresholds
        ])
        best_cutoff = float(thresholds[int(np.argmax(train_f1))])
        # Fit the SAME model once; only decision rule changes on outer fold.
        model = clone(candidate).fit(Xtr, ytr)
        p_valid = model.predict_proba(Xva)[:, 1]
        default = metrics(yva, p_valid, 0.50)
        tuned = metrics(yva, p_valid, best_cutoff)
        rows.append(dict(
            fold=fold, model=args.model, threshold_from_train=best_cutoff,
            training_oof_best_f1=float(train_f1.max()),
            default_f1=default['f1'], tuned_f1=tuned['f1'],
            f1_difference=tuned['f1'] - default['f1'],
            default_precision=default['precision'], tuned_precision=tuned['precision'],
            default_recall=default['recall'], tuned_recall=tuned['recall'],
            average_precision=default['average_precision'],
        ))
        print(f'Fold {fold}/{n_outer}: cutoff={best_cutoff:.3f} '
              f'F1@0.50={default["f1"]:.4f} '
              f'F1@tuned={tuned["f1"]:.4f} '
              f'diff={tuned["f1"] - default["f1"]:+.4f}', flush=True)

    result = pd.DataFrame(rows)
    cols = ['default_f1', 'tuned_f1', 'f1_difference', 'default_precision',
            'tuned_precision', 'default_recall', 'tuned_recall', 'average_precision']
    print('\n=== MATCHED OUTER-FOLD AVERAGES ===')
    print(result[cols].mean().round(4).to_string())
    positive = int((result['f1_difference'] > 0).sum())
    ties = int((result['f1_difference'] == 0).sum())
    print(f'Folds improved by threshold tuning: {positive}/{n_outer}; ties={ties}')
    print(f'Chosen thresholds: {result["threshold_from_train"].round(3).tolist()}')
    print('Each comparison uses identical validation predictions; only cutoff differs.')
    print('Variation between folds and previous exploration limit any improvement claim.')

    file_out = Path(__file__).resolve().parent / f'threshold_stability_{args.model}_{args.mode}_results.csv'
    result.to_csv(file_out, index=False)
    print('Saved:', file_out)


if __name__ == '__main__':
    main()
