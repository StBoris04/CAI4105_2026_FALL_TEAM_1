"""Challenge 2: reproducible outer-fold audit of the TEAM-INSPIRED selection pipeline.

Place in challenge2/models/ alongside:
  compare_team_features.py, team_pipeline_audit.py, compare_fixed_vs_auto.py

From the repository root:
  python challenge2/models/consolidated_pipeline_validation.py --mode quick
  python challenge2/models/consolidated_pipeline_validation.py --mode full

Quick: 4 models and reduced search grids; NOT the complete team configuration.
Full: the original 8 model families and tuning grids; potentially very expensive.

Compares a training-selected candidate, a fixed equal-weight 3-tree ensemble,
and a fixed Random Forest using the SAME outer validation folds. Hyperparameter
search, candidate selection, and threshold tuning take place ONLY in each outer
TRAINING fold. The hidden test file is NEVER read, and no submission is written.

This is a separate reimplementation and is not a direct execution of
purchase_prediction.py. The labeled dataset has already been extensively
explored; all reported results remain exploratory, not independent proof.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV

from compare_team_features import locate_data
from compare_fixed_vs_auto import get_ensemble
from team_pipeline_audit import choose_team_candidate, search_specs, team_pipe

MODEL_NAMES = ['Automatic Selection', 'Fixed Soft-Voting Ensemble', 'Fixed Random Forest']


def read_training_data(path):
    if not path.is_file():
        raise FileNotFoundError(f'Training file not found: {path}')
    data = pd.read_csv(path)
    expected = {'ID', 'Revenue', 'Administrative', 'Informational', 'ProductRelated',
                'Administrative_Duration', 'Informational_Duration',
                'ProductRelated_Duration', 'PageValues', 'BounceRates', 'ExitRates',
                'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region',
                'TrafficType', 'VisitorType', 'Weekend'}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data.empty or data['Revenue'].isna().any():
        raise ValueError('Training data is empty or Revenue contains null values.')
    if data['ID'].isna().any() or not data['ID'].is_unique:
        raise ValueError('Training IDs must be non-null and unique.')
    # Pandas reads true/false literals as booleans; numeric 0/1 is also valid.
    if not set(data['Revenue'].unique()).issubset({True, False, 0, 1}):
        raise ValueError('Revenue must contain only True/False or 0/1 values.')
    y = data['Revenue'].astype(bool)
    if y.nunique() != 2:
        raise ValueError('Both target classes must be present.')
    X = data.drop(columns=['ID', 'Revenue'])
    return X, y


def evaluate(name, model, X_train, y_train, X_valid, y_valid, cv):
    # TunedThresholdClassifierCV refits the selected estimator on all outer
    # training rows after its inner threshold search. Outer labels are unseen.
    tuned = TunedThresholdClassifierCV(
        clone(model), scoring='f1', cv=cv, thresholds=80, n_jobs=1,
    )
    tuned.fit(X_train, y_train)
    probabilities = tuned.predict_proba(X_valid)[:, 1]
    predictions = tuned.predict(X_valid)
    return {
        'approach': name,
        'f1': float(f1_score(y_valid, predictions, zero_division=0)),
        'precision': float(precision_score(y_valid, predictions, zero_division=0)),
        'recall': float(recall_score(y_valid, predictions, zero_division=0)),
        'average_precision': float(average_precision_score(y_valid, probabilities)),
        'threshold': float(tuned.best_threshold_),
        'tp': int(((predictions == 1) & (y_valid == 1)).sum()),
        'fp': int(((predictions == 1) & (y_valid == 0)).sum()),
        'tn': int(((predictions == 0) & (y_valid == 0)).sum()),
        'fn': int(((predictions == 0) & (y_valid == 1)).sum()),
    }


def markdown_report(df, metadata):
    metrics = ['f1', 'precision', 'recall', 'average_precision']
    means = df.groupby('approach')[metrics].mean().reindex(MODEL_NAMES)
    wide = df.pivot(index='fold', columns='approach', values='f1')
    differences = wide['Automatic Selection'] - wide['Fixed Soft-Voting Ensemble']
    out = [
        '# Challenge 2 — Consolidated Pipeline Validation',
        '',
        f"**Run (UTC):** {metadata['timestamp_utc']}",
        f"**Mode:** {metadata['mode']} | **Outer folds:** {metadata['outer_folds']} | "
        f"**Inner folds:** {metadata['inner_folds']}",
        f"**Rows:** {metadata['training_rows']} | **Positive rate:** {metadata['positive_rate']:.2%}",
        f"**Training CSV SHA256:** `{metadata['train_sha256']}`",
        f"**Environment:** Python {metadata['python']} / scikit-learn {metadata['sklearn']}",
        '',
        '## Matched outer-fold results',
        '',
        '| Approach | Mean F1 | Mean precision | Mean recall | Mean AP |',
        '|---|---:|---:|---:|---:|',
    ]
    for approach, row in means.iterrows():
        out.append(f"| {approach} | {row.f1:.4f} | {row.precision:.4f} | "
                   f"{row.recall:.4f} | {row.average_precision:.4f} |")
    out += [
        '',
        '## Per-fold scores',
        '',
        '| Fold | Automatically selected model | Auto F1 | Ensemble F1 | Random Forest F1 | Auto − ensemble |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for fold in wide.index:
        selected = df[(df.fold == fold) & (df.approach == 'Automatic Selection')]['selected_model'].iloc[0]
        out.append(f"| {fold} | {selected} | {wide.loc[fold, 'Automatic Selection']:.4f} | "
                   f"{wide.loc[fold, 'Fixed Soft-Voting Ensemble']:.4f} | "
                   f"{wide.loc[fold, 'Fixed Random Forest']:.4f} | {differences.loc[fold]:+.4f} |")
    out += [
        '',
        f"**Mean paired Auto − ensemble F1:** {differences.mean():+.4f}; "
        f"auto won {(differences > 0).sum()}/{len(differences)} folds; "
        f"tied {(differences == 0).sum()}/{len(differences)}.",
        '',
        '## Method and interpretation',
        '',
        '- All three approaches were evaluated on exactly the same outer validation rows.',
        '- Hyperparameter tuning, candidate ranking, and threshold tuning used only the outer training rows.',
        '- The ensemble uses tuned Random Forest, Gradient Boosting and HistGradientBoosting members.',
        '- The automatic selection is the team-inspired audit implementation; it does **not** execute the original script.',
        '- Quick mode uses reduced grids; full mode uses the eight original model families and grids, but is still a reimplementation.',
        '- The test CSV is never read, no submission is produced, and the original team source is unchanged.',
        '- This training dataset and the candidate recipes have been inspected repeatedly. Report these as exploratory results, **not proof** of a new improvement or a final unbiased score.',
        '- The original team code performs hyperparameter/model/threshold selection with a different detailed sequence; full equivalence has not been established.',
        '',
        '## Suggested action',
        '',
        'Keep the production model unchanged unless a meaningful advantage is repeatable and the team reviews it. '
        'If the scores are similar, prefer the simpler and more reproducible choice.',
        '',
    ]
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument('--train-data', type=Path, default=None,
                        help='Optional training CSV path; for testing only.')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.outer_folds < 2:
        parser.error('--outer-folds must be 2 or more')
    path = args.train_data or locate_data()
    X, y = read_training_data(path)
    inner_folds = 2 if args.mode == 'quick' else 5
    min_count = int(y.value_counts().min())
    if min_count < max(args.outer_folds, 8 * inner_folds):
        parser.error('Not enough examples of the minority class for nested validation')
    specs = search_specs(args.mode)
    if not {'Random Forest', 'Gradient Boosting', 'HistGradientBoosting', 'Logistic Regression'}.issubset(specs):
        raise RuntimeError('Team candidate models missing from search specs')
    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=2027)
    cutoffs = np.arange(0.10, 0.901, 0.025) if args.mode == 'quick' else np.arange(0.05, 0.951, 0.005)
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        mode=args.mode, outer_folds=args.outer_folds, inner_folds=inner_folds,
        training_rows=len(y), positive_rate=float(y.mean()),
        train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__,
        models=list(specs), random_state_outer=2027,
        note='Exploratory only. Dataset and recipes already investigated.',
    )
    print('=== CONSOLIDATED CHALLENGE 2 VALIDATION ===', flush=True)
    print(f'Mode={args.mode} rows={len(y)} outer={args.outer_folds} inner={inner_folds}', flush=True)
    print('No hidden test file read; model choice, grids and threshold stay inside outer training.', flush=True)
    if args.mode == 'full':
        print('WARNING: full grids evaluate many combinations and can be very expensive.', flush=True)

    rows = []
    for fold, (fit_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        Xtr, ytr = X.iloc[fit_idx], y.iloc[fit_idx]
        Xval, yval = X.iloc[val_idx], y.iloc[val_idx]
        cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42 + fold)
        fitted = {}
        print(f'\nOuter fold {fold}/{args.outer_folds}:', flush=True)
        for name, (learner, grid) in specs.items():
            search = GridSearchCV(team_pipe(name, learner, Xtr), grid,
                                  scoring='f1', cv=cv, n_jobs=1,
                                  error_score='raise', refit=True)
            search.fit(Xtr, ytr)
            fitted[name] = search
            print(f'  {name:22s} inner grid F1={search.best_score_:.4f}', flush=True)

        selected_name, auto_estimator, _, _ = choose_team_candidate(
            fitted, Xtr, ytr, cv, cutoffs, inner_folds,
        )
        approaches = [
            ('Automatic Selection', auto_estimator),
            ('Fixed Soft-Voting Ensemble', get_ensemble(fitted, Xtr)),
            ('Fixed Random Forest', fitted['Random Forest'].best_estimator_),
        ]
        for name, estimator in approaches:
            row = evaluate(name, estimator, Xtr, ytr, Xval, yval, cv)
            row.update(fold=fold, selected_model=selected_name)
            rows.append(row)
            print(f'  {name:27s} F1={row["f1"]:.4f} P={row["precision"]:.4f} '
                  f'R={row["recall"]:.4f} threshold={row["threshold"]:.3f}', flush=True)
        print(f'  Training-only selection: {selected_name}', flush=True)

    result = pd.DataFrame(rows)
    means = result.groupby('approach')[['f1', 'precision', 'recall', 'average_precision']].mean().reindex(MODEL_NAMES)
    print('\n=== MATCHED OUTER VALIDATION MEANS ===\n' + means.round(4).to_string(), flush=True)
    output_dir = args.output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / f'consolidated_pipeline_{args.mode}_results.csv'
    report_file = output_dir / f'consolidated_pipeline_{args.mode}_report.md'
    config_file = output_dir / f'consolidated_pipeline_{args.mode}_config.json'
    result.to_csv(csv_file, index=False)
    report_file.write_text(markdown_report(result, metadata), encoding='utf-8')
    config_file.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('\nSaved: ' + str(csv_file), flush=True)
    print('Saved: ' + str(report_file), flush=True)
    print('Saved: ' + str(config_file), flush=True)
    print('This is a team-inspired evaluation, not a claim of full production equivalence.', flush=True)
    print('Results are exploratory due to repeated investigation of the dataset.', flush=True)


if __name__ == '__main__':
    main()
