"""Compare tuned Random Forests on team vs proposed feature engineering.

Put this script beside compare_team_features.py in challenge2/models/.
Uses challenge2_train.csv only; never reads challenge2_test.csv.
Compatible with Python 3.11 and scikit-learn 1.5.2.

Run a smaller exploratory comparison first:
    python challenge2/models/tuned_random_forest_comparison.py --mode quick

Optionally use the team's exact Random Forest hyperparameter grid (slower):
    python challenge2/models/tuned_random_forest_comparison.py --mode full

Each OUTER validation fold is excluded from all hyperparameter and threshold
selection. Results are exploratory because these feature sets were previously
examined. This is NOT a replication of the team's full multi-model ensemble.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict,
)

from compare_team_features import locate_data, pipeline


def choose_threshold(y_true, prob):
    """Choose a threshold without using the outer validation fold."""
    thresholds = np.round(np.arange(0.15, 0.751, 0.025), 3)
    scores = [f1_score(y_true, prob >= t, zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def get_settings(mode):
    if mode == "full":
        # This is the exact Random Forest grid and forest size from the team's
        # provided purchase_prediction.py (but NOT its entire final pipeline).
        return {
            "n_estimators": 300,
            "inner_folds": 5,
            "params": {
                "model__max_features": ["sqrt", 0.3],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__class_weight": [None, "balanced_subsample"],
            },
        }
    # Smaller preview to verify the setup and see if tuning is promising.
    # Do not describe quick mode as an exact reproduction of the team's grid.
    return {
        "n_estimators": 100,
        "inner_folds": 2,
        "params": {
            "model__max_features": ["sqrt", 0.3],
            "model__min_samples_leaf": [1, 3],
            "model__class_weight": [None],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--outer-folds", type=int, default=3,
                        help="Number of matched outer validation folds (default: 3).")
    args = parser.parse_args()
    if args.outer_folds < 2:
        parser.error("--outer-folds must be at least 2")

    data_path = locate_data()
    df = pd.read_csv(data_path).dropna(subset=["Revenue"])
    X = df.drop(columns=["Revenue", "ID"], errors="ignore")
    y = df["Revenue"].astype(bool)
    settings = get_settings(args.mode)
    if y.nunique() != 2 or y.value_counts().min() < max(args.outer_folds, settings["inner_folds"]):
        parser.error("Not enough rows in both classes for cross-validation")

    recipes = ["Team features", "New features"]
    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=2026)
    output_rows = []
    n_combinations = int(np.prod([len(v) for v in settings["params"].values()]))

    print("\n=== TUNED RANDOM FOREST: MATCHED FEATURE COMPARISON ===", flush=True)
    print(f"Mode: {args.mode}; training data: {data_path}; rows: {len(df)}", flush=True)
    print(f"Outer folds: {args.outer_folds}; inner folds: {settings['inner_folds']}; "
          f"trees per forest: {settings['n_estimators']}; settings per recipe: {n_combinations}", flush=True)
    print("All hyperparameters and thresholds are learned from outer TRAINING rows only.", flush=True)
    print("Hidden test CSV and the team's purchase_prediction.py are never opened or changed.", flush=True)
    print("Feature recipes have been explored previously; results are exploratory.\n", flush=True)

    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        inner_cv = StratifiedKFold(n_splits=settings["inner_folds"], shuffle=True,
                                   random_state=6000 + fold)
        for feature_name in recipes:
            print(f"Fold {fold}/{args.outer_folds}: tuning {feature_name}...", flush=True)
            estimator = pipeline("Random Forest", feature_name, X)
            estimator.set_params(model__n_estimators=settings["n_estimators"], model__n_jobs=1)
            search = GridSearchCV(
                estimator, settings["params"], cv=inner_cv, scoring="f1",
                refit=True, n_jobs=1, error_score="raise",
            )
            search.fit(X_train, y_train)

            # Generate probabilities on the OUTER TRAINING data using inner-CV.
            # Hyperparameters were selected on this outer training data, so
            # threshold tuning is exploratory; outer validation remains held out.
            train_oof_prob = cross_val_predict(
                search.best_estimator_, X_train, y_train, cv=inner_cv,
                method="predict_proba", n_jobs=1,
            )[:, 1]
            threshold, train_oof_f1 = choose_threshold(y_train, train_oof_prob)

            valid_prob = search.best_estimator_.predict_proba(X_valid)[:, 1]
            pred = valid_prob >= threshold
            row = {
                "fold": fold,
                "features": feature_name,
                "f1": f1_score(y_valid, pred, zero_division=0),
                "precision": precision_score(y_valid, pred, zero_division=0),
                "recall": recall_score(y_valid, pred, zero_division=0),
                "average_precision": average_precision_score(y_valid, valid_prob),
                "threshold": threshold,
                "inner_cv_f1_at_0_5": float(search.best_score_),
                "training_oof_f1_at_threshold": train_oof_f1,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "mode": args.mode,
            }
            output_rows.append(row)
            print(f"  {feature_name:14s}: outer F1={row['f1']:.4f}, "
                  f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
                  f"threshold={threshold:.3f}", flush=True)

    results = pd.DataFrame(output_rows)
    pairs = results.pivot(index="fold", columns="features", values="f1")
    pairs["new_minus_team_f1"] = pairs["New features"] - pairs["Team features"]
    print("\n=== AVERAGE OUTER-FOLD METRICS ===")
    print(results.groupby("features")[["f1", "precision", "recall", "average_precision"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("\n=== PAIRED F1 CHANGE: NEW MINUS TEAM ===")
    print(pairs.round(4).to_string())
    difference = pairs["new_minus_team_f1"]
    print(f"Mean F1 change: {difference.mean():+.4f}; "
          f"new features win {int((difference > 0).sum())}/{len(difference)} folds")
    print("These outer folds are not independent proof of an improvement, and "
          "this does not evaluate the team's entire model-selection workflow.")

    out_dir = Path(__file__).resolve().parent
    detail_path = out_dir / f"tuned_forest_{args.mode}_results.csv"
    paired_path = out_dir / f"tuned_forest_{args.mode}_paired.csv"
    results.to_csv(detail_path, index=False)
    pairs.reset_index().to_csv(paired_path, index=False)
    print(f"Saved: {detail_path}\nSaved: {paired_path}", flush=True)


if __name__ == "__main__":
    main()
