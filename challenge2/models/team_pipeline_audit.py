"""Audit the team's Challenge 2 model-SELECTION workflow with outer validation.

Put this file in challenge2/models/ beside compare_team_features.py.
Python 3.11 + scikit-learn 1.5.2 compatible.

Run a *small* version of the team's selection workflow first:
    python challenge2/models/team_pipeline_audit.py --mode quick

Full mode uses the team's 8 original hyperparameter grids and a 5-fold
inner search. It may be extremely compute-intensive:
    python challenge2/models/team_pipeline_audit.py --mode full

The workflow selects hyperparameters, candidate models, and prediction
thresholds WITHOUT access to each outer validation fold. It uses only
challenge2_train.csv, never challenge2_test.csv or predictions.csv.

This is a standalone REIMPLEMENTATION for auditing, not execution of the
team's purchase_prediction.py. It does not modify the team's model.
Quick mode uses reduced grids and is NOT an exact reproduction of the full
configuration. This is exploratory: we have already investigated this data.
"""

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from compare_team_features import locate_data, pipeline

SEED = 42


def search_specs(mode):
    """Model grids adapted from the user's provided purchase_prediction.py."""
    if mode == 'quick':
        # Representative shortlist from the team's final candidate pool;
        # NOT its full model search or original tree counts.
        return {
            'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=SEED,
                                                     n_jobs=1), {
                'model__max_features': ['sqrt'],
                'model__min_samples_leaf': [1, 3],
                'model__class_weight': [None],
            }),
            'Gradient Boosting': (GradientBoostingClassifier(
                n_estimators=100, subsample=0.8, random_state=SEED), {
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [2],
            }),
            'HistGradientBoosting': (HistGradientBoostingClassifier(
                max_iter=120, l2_regularization=1.0, random_state=SEED), {
                'model__learning_rate': [0.03, 0.1],
                'model__max_leaf_nodes': [15],
                'model__min_samples_leaf': [20],
                'model__class_weight': [None],
            }),
            'Logistic Regression': (LogisticRegression(max_iter=1500,
                                                      random_state=SEED), {
                'model__C': [0.1, 1],
                'model__class_weight': [None],
            }),
        }
    # Original team grids and base configurations from purchase_prediction.py.
    return {
        'Logistic Regression': (LogisticRegression(max_iter=5000, random_state=SEED), {
            'model__C': [0.01, 0.03, 0.1, 0.3, 1, 3],
            'model__class_weight': [None, 'balanced'],
        }),
        'Naive Bayes': (GaussianNB(), {
            'model__var_smoothing': np.logspace(-9, -1, 9),
        }),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {
            'model__n_neighbors': [5, 11, 21, 41],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2],
        }),
        'Support Vector Machine': (SVC(random_state=SEED), {
            'model__C': [0.3, 1, 3],
            'model__gamma': ['scale', 0.01],
            'model__class_weight': ['balanced'],
        }),
        'Decision Tree': (DecisionTreeClassifier(random_state=SEED), {
            'model__max_depth': [3, 5, 7, 10],
            'model__min_samples_leaf': [5, 20, 50],
            'model__class_weight': [None, 'balanced'],
        }),
        'Random Forest': (RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                 n_jobs=1), {
            'model__max_features': ['sqrt', 0.3],
            'model__min_samples_leaf': [1, 3, 5, 10],
            'model__class_weight': [None, 'balanced_subsample'],
        }),
        'Gradient Boosting': (GradientBoostingClassifier(subsample=0.8,
                                                        random_state=SEED), {
            'model__n_estimators': [150, 300],
            'model__learning_rate': [0.03, 0.1],
            'model__max_depth': [2, 3, 4],
        }),
        'HistGradientBoosting': (HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, random_state=SEED), {
            'model__learning_rate': [0.03, 0.1],
            'model__max_leaf_nodes': [15, 31],
            'model__min_samples_leaf': [20, 50],
            'model__class_weight': [None, 'balanced'],
        }),
    }


def team_pipe(name, model, X):
    """Same preprocessing + feature engineering recipe as the team."""
    result = pipeline('Random Forest', 'Team features', X)
    return result.set_params(model=clone(model))


def choose_team_candidate(fitted, Xtrain, ytrain, inner_cv, thresholds,
                          threshold_cv_folds):
    """Select among ALL five candidates using only outer-training examples.

    Keep the existing signature because compare_fixed_vs_auto.py and
    consolidated_pipeline_validation.py already import this function.

    The passed GridSearchCV objects were tuned within the outer training fold.
    Candidate comparison also happens within that fold; then the caller
    evaluates the winner on the untouched outer validation fold.

    Candidate CV scores are selection diagnostics, not independent estimates:
    the hyperparameters were selected from these same outer-training rows.
    """
    base_names = ['Random Forest', 'Gradient Boosting', 'HistGradientBoosting']
    votes = VotingClassifier(
        estimators=[(name.lower().replace(' ', '_'),
                     clone(fitted[name].best_estimator_.named_steps['model']))
                    for name in base_names], voting='soft', n_jobs=1,
    )
    options = {name: clone(fitted[name].best_estimator_) for name in base_names}
    options['Soft-Voting Ensemble'] = team_pipe('Soft-Voting Ensemble', votes, Xtrain)
    options['Logistic Regression'] = clone(
        fitted['Logistic Regression'].best_estimator_)

    # Do not prefilter using thresholds fitted against these same training
    # targets. Every candidate receives the same inner selection procedure.
    selection_cv = StratifiedKFold(
        n_splits=threshold_cv_folds, shuffle=True, random_state=SEED + 17)
    candidate_cv = []
    for name, estimator in options.items():
        threshold_estimator = TunedThresholdClassifierCV(
            clone(estimator), scoring='f1', cv=threshold_cv_folds,
            thresholds=80, n_jobs=1)
        scores = cross_val_score(
            threshold_estimator, Xtrain, ytrain, cv=selection_cv,
            scoring='f1', n_jobs=1, error_score='raise')
        candidate_cv.append((name, float(np.mean(scores)), float(np.std(scores))))
    candidate_cv.sort(key=lambda item: item[1], reverse=True)

    winner = candidate_cv[0][0]
    # Keep the team's rule favoring a simpler model if the ensemble advantage
    # is smaller than half its observed candidate-CV standard deviation.
    if winner == 'Soft-Voting Ensemble':
        singles = [row for row in candidate_cv if row[0] != 'Soft-Voting Ensemble']
        if singles and candidate_cv[0][1] - singles[0][1] < candidate_cv[0][2] / 2:
            winner = singles[0][0]

    # Third return value is retained for compatibility with other scripts.
    ranking = [(name, score) for name, score, _ in candidate_cv]
    return winner, options[winner], ranking, candidate_cv

def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--mode', choices=['quick', 'full'], default='quick')
    arg.add_argument('--outer-folds', type=int, default=3)
    arg.add_argument('--only-fold', type=int, default=None,
                     help='Run only this outer fold (1 through --outer-folds); useful for full grids.')
    args = arg.parse_args()
    if args.outer_folds < 2:
        arg.error('--outer-folds must be at least 2')
    if args.only_fold is not None and not (1 <= args.only_fold <= args.outer_folds):
        arg.error('--only-fold must be between 1 and --outer-folds')

    path = locate_data()
    data = pd.read_csv(path).dropna(subset=['Revenue'])
    X = data.drop(columns=['Revenue', 'ID'], errors='ignore')
    y = data['Revenue'].astype(bool)
    inner_folds = 2 if args.mode == 'quick' else 5
    selection_folds = 2 if args.mode == 'quick' else 5
    required_classes = max(args.outer_folds, inner_folds, selection_folds ** 2)
    if y.nunique() != 2 or y.value_counts().min() < required_classes:
        arg.error('Not enough examples per class for selected CV folds')

    specs = search_specs(args.mode)
    thresholds = np.round(np.arange(0.1, 0.901, 0.025), 3) if args.mode == 'quick' \
        else np.round(np.arange(0.05, 0.951, 0.005), 3)
    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True,
                               random_state=2027)

    print('\n=== TEAM PIPELINE AUDIT (nested outer-fold evaluation) ===', flush=True)
    print('Mode:', args.mode, '| Rows:', len(data), '| Positive rate:',
          f'{y.mean():.1%}', flush=True)
    print('Outer folds:', args.outer_folds, '| inner folds:', inner_folds,
          '| models tuned:', ', '.join(specs), flush=True)
    print('Quick mode is an approximation, not the complete team grid.' if args.mode == 'quick'
          else 'Full mode uses the original team hyperparameter grids.', flush=True)
    print('All selection happens within each outer training fold.', flush=True)
    print('The hidden test dataset is never read and no submission is generated.', flush=True)
    print('The team source file is not executed or modified.', flush=True)
    if args.only_fold is not None:
        print(f'Running only outer fold {args.only_fold}/{args.outer_folds}; results are PARTIAL.', flush=True)
    print('', flush=True)

    rows = []
    for outer_index, (tr_idx, va_idx) in enumerate(outer_cv.split(X, y), 1):
        if args.only_fold is not None and outer_index != args.only_fold:
            continue
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                   random_state=SEED + outer_index)
        print(f'Outer fold {outer_index}/{args.outer_folds}: {len(Xtr)} train, '
              f'{len(Xva)} validation', flush=True)
        fits = {}
        start = time.time()
        for name, (model, grid) in specs.items():
            search = GridSearchCV(
                team_pipe(name, model, Xtr), grid, scoring='f1',
                cv=inner_cv, refit=True, n_jobs=1, error_score='raise',
            )
            search.fit(Xtr, ytr)
            fits[name] = search
            print(f'    tuned {name:23s} inner F1={search.best_score_:.4f}', flush=True)
        selected, estimator, ranking, nested = choose_team_candidate(
            fits, Xtr, ytr, inner_cv, thresholds, selection_folds)
        final = TunedThresholdClassifierCV(clone(estimator), scoring='f1',
                                           cv=inner_cv, thresholds=100,
                                           n_jobs=1)
        final.fit(Xtr, ytr)
        pred = final.predict(Xva)
        prob = final.predict_proba(Xva)[:, 1]
        row = {
            'fold': outer_index,
            'selected_model': selected,
            'f1': f1_score(yva, pred, zero_division=0),
            'precision': precision_score(yva, pred, zero_division=0),
            'recall': recall_score(yva, pred, zero_division=0),
            'average_precision': average_precision_score(yva, prob),
            'roc_auc': roc_auc_score(yva, prob),
            'threshold': float(final.best_threshold_),
            'selection_cv_f1': next(s[1] for s in nested if s[0] == selected),
            'candidate_selection_ranking': json.dumps(ranking),
            'candidate_selection_cv_scores': json.dumps(nested),
            'best_grids': json.dumps({name: {k: (v.item() if isinstance(v, np.generic) else v)
                                                  for k,v in search.best_params_.items()}
                                      for name, search in fits.items()}, sort_keys=True),
            'mode': args.mode,
        }
        rows.append(row)
        print('    candidate CV:', ', '.join(f'{n}={score:.4f}' for n, score, _ in nested), flush=True)
        print(f"  -> selected {selected} | OUTER F1={row['f1']:.4f}, "
              f"P={row['precision']:.4f}, R={row['recall']:.4f}, "
              f"threshold={row['threshold']:.3f} | {time.time()-start:.0f}s\n", flush=True)

    result = pd.DataFrame(rows)
    print('=== UNSEEN OUTER-FOLD METRICS (WITHIN THIS RUN) ===', flush=True)
    print(result[['fold', 'selected_model', 'f1', 'precision', 'recall',
                  'average_precision', 'threshold']].round(4).to_string(index=False))
    print('\nMean metrics:', result[['f1', 'precision', 'recall', 'average_precision']]
          .mean().round(4).to_dict(), flush=True)
    print('The folds are new for the script, but the dataset has already been explored;',
          'these are exploratory results, not independent final proof.', flush=True)
    print('Model selection tested all five candidates on each evaluated outer-training fold.', flush=True)
    if args.only_fold is not None:
        print('PARTIAL RESULT: do not report this fold as the full cross-validation mean.', flush=True)
    print('Do NOT compare this mean F1 directly with earlier scores from other splits.',
          flush=True)

    suffix = f'_fold{args.only_fold}' if args.only_fold is not None else ''
    out_path = Path(__file__).resolve().parent / f'team_pipeline_audit_{args.mode}{suffix}_results.csv'
    result.to_csv(out_path, index=False)
    print('Saved:', out_path, flush=True)


if __name__ == '__main__':
    main()
