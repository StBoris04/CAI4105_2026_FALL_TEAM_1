"""Repeated cross-validation comparing the team's vs proposed feature engineering.

Put this file alongside compare_team_features.py in challenge2/models/.
Run using the same Python 3.11 virtual environment you used previously.

This script imports only helper functions from compare_team_features.py.
It NEVER runs purchase_prediction.py, uses challenge2_test.csv, or alters either.

Evaluation per outer fold:
  * Create a stratified 20% threshold-tuning split from the outer TRAINING fold.
  * Train each pipeline on the remaining outer training rows.
  * Pick its probability threshold using only the threshold-tuning split.
  * Score its predictions ONCE on the outer validation fold.
Team and new features use the same folds and identical learner settings.

Note: This is a repeated-CV *exploratory* comparison, not independent proof of
improvement over the team's fully tuned model. Splits overlap across repeats.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from compare_team_features import locate_data, pipeline


def main():
    parser = argparse.ArgumentParser(description="Compare team vs new features using repeated CV.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model", choices=["both", "forest", "hist"], default="both",
                        help="Run both algorithms (default), forest only, or hist only.")
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1:
        parser.error("--folds must be at least 2; --repeats must be at least 1")

    data_path = locate_data()
    df = pd.read_csv(data_path).dropna(subset=["Revenue"])
    X = df.drop(columns=["ID", "Revenue"], errors="ignore")
    y = df["Revenue"].astype(bool)
    if y.nunique() != 2 or y.value_counts().min() < args.folds:
        parser.error("Not enough observations per class for the requested number of folds")

    learners = ({"forest": ["Random Forest"], "hist": ["HistGradientBoosting"],
                 "both": ["Random Forest", "HistGradientBoosting"]}[args.model])
    feature_names = ("Team features", "New features")
    thresholds = np.round(np.arange(0.15, 0.651, 0.025), 3)
    splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                      random_state=2026)
    rows = []

    print("Repeated stratified cross-validation: team vs new features")
    print(f"Data: {data_path} | rows={len(df)} | positive rate={y.mean():.1%}")
    print(f"Folds: {args.folds} | Repeats: {args.repeats} | Model(s): {', '.join(learners)}")
    print("The SAME outer validation rows are used for every matched feature comparison.")
    print("Thresholds are chosen using only data within each outer TRAINING fold.")
    print("The hidden challenge2_test.csv is NEVER used.")
    print("This does NOT evaluate the team's complete grid-search/ensemble system.\n", flush=True)

    for split_number, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        repeat_number = (split_number - 1) // args.folds + 1
        fold_number = (split_number - 1) % args.folds + 1
        X_outer, y_outer = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        # One matched threshold-tuning split per outer fold for ALL candidates.
        X_fit, X_tune, y_fit, y_tune = train_test_split(
            X_outer, y_outer, test_size=0.20, stratify=y_outer,
            random_state=5000 + split_number)

        for learner in learners:
            fold_scores = {}
            for feature_name in feature_names:
                model = pipeline(learner, feature_name, X)
                model.fit(X_fit, y_fit)
                tuning_prob = model.predict_proba(X_tune)[:, 1]
                tuning_scores = np.asarray([
                    f1_score(y_tune, tuning_prob >= threshold, zero_division=0)
                    for threshold in thresholds
                ])
                selected_threshold = float(thresholds[int(tuning_scores.argmax())])
                pred = model.predict_proba(X_valid)[:, 1] >= selected_threshold
                row = {
                    "repeat": repeat_number, "fold": fold_number, "split": split_number,
                    "model": learner, "features": feature_name,
                    "threshold": selected_threshold,
                    "f1": f1_score(y_valid, pred, zero_division=0),
                    "precision": precision_score(y_valid, pred, zero_division=0),
                    "recall": recall_score(y_valid, pred, zero_division=0),
                }
                rows.append(row)
                fold_scores[feature_name] = row["f1"]
            difference = fold_scores["New features"] - fold_scores["Team features"]
            print(f"Repeat {repeat_number}/{args.repeats}, fold {fold_number}/{args.folds} | "
                  f"{learner:20s} team={fold_scores['Team features']:.4f} "
                  f"new={fold_scores['New features']:.4f} difference={difference:+.4f}", flush=True)

    results = pd.DataFrame(rows)
    # Paired differences must match the SAME model / repeat / fold.
    paired = results.pivot(index=["model", "repeat", "fold", "split"],
                           columns="features", values="f1").reset_index()
    paired["new_minus_team_f1"] = paired["New features"] - paired["Team features"]

    print("\n=== MEAN VALIDATION METRICS ACROSS OUTER FOLDS ===")
    summary = (results.groupby(["model", "features"])[["f1", "precision", "recall"]]
               .agg(["mean", "std"]).round(4))
    print(summary.to_string())
    print("\n=== PAIRED F1 DIFFERENCES: NEW MINUS TEAM ===")
    for learner in learners:
        diffs = paired.loc[paired["model"] == learner, "new_minus_team_f1"]
        print(f"{learner}: mean={diffs.mean():+.4f}, std={diffs.std():.4f}, "
              f"new wins={int((diffs > 0).sum())}/{len(diffs)} splits, "
              f"ties={int((diffs == 0).sum())}")
    print("\nReminder: repeated folds overlap; these are not independent trials, "
          "and small differences can reflect sampling variability.")
    print("The team’s FULL optimized model has not yet been compared.")

    output_dir = Path(__file__).resolve().parent
    detail_path = output_dir / "repeated_feature_validation_results.csv"
    paired_path = output_dir / "repeated_feature_validation_differences.csv"
    results.to_csv(detail_path, index=False)
    paired.to_csv(paired_path, index=False)
    print(f"Saved fold results: {detail_path}")
    print(f"Saved paired differences: {paired_path}")


if __name__ == "__main__":
    main()
