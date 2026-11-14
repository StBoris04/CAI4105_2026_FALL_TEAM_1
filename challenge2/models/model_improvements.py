r"""Independent Challenge 2 model experiment; does not modify the team model.

Run from the project folder with:
    .\.venv311\Scripts\python.exe .\CAI4105_2026_FALL_TEAM_1-main\challenge2\models\model_improvements.py

Uses challenge2_train.csv only. Never reads challenge2_test.csv.
Compatible with Python 3.11 and scikit-learn 1.5.2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Revenue"
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]


def locate_training_file():
    """Find data from either a nested team repo or a direct challenge2 folder."""
    here = Path(__file__).resolve()
    candidates = []
    for parent in here.parents:
        candidates.extend([
            parent / "data" / "challenge2_train.csv",
            parent / "challenge2" / "data" / "challenge2_train.csv",
            parent / "challenge2_train.csv",
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Can't find challenge2_train.csv. Expected it inside challenge2/data/. "
        "Make sure you placed this script in challenge2/models/."
    )


def add_behavior_features(frame):
    """Row-only feature engineering; no information from other rows or labels."""
    d = frame.copy()
    pages = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    duration = (
        d["Administrative_Duration"]
        + d["Informational_Duration"]
        + d["ProductRelated_Duration"]
    )
    d["TotalPages"] = pages
    d["TotalDuration"] = duration
    d["AvgDurationPerPage"] = duration / (pages + 1)
    d["ProductPageShare"] = d["ProductRelated"] / (pages + 1)
    d["ProductDurationShare"] = d["ProductRelated_Duration"] / (duration + 1)
    d["ExitMinusBounce"] = d["ExitRates"] - d["BounceRates"]
    d["PageValuePerProductPage"] = d["PageValues"] / (d["ProductRelated"] + 1)
    d["ProductDurationPerPage"] = d["ProductRelated_Duration"] / (d["ProductRelated"] + 1)
    d["HasPageValue"] = (d["PageValues"] > 0).astype(int)
    return d


def make_preprocessor(feature_columns, scaled):
    categorical = [c for c in CATEGORICAL if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scaled:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
        ("numeric", Pipeline(numeric_steps), numeric),
    ])


def make_model(name, raw_columns, engineered_columns):
    engineered = name.endswith("+ new features")
    features = engineered_columns if engineered else raw_columns
    if name.startswith("Logistic"):
        learner = LogisticRegression(max_iter=3000, random_state=SEED)
        scale = True
    elif name.startswith("Random Forest"):
        learner = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
        scale = False
    else:
        learner = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=SEED,
        )
        scale = False
    steps = []
    if engineered:
        steps.append(("new_features", FunctionTransformer(add_behavior_features, validate=False)))
    steps += [("preprocess", make_preprocessor(features, scale)), ("model", learner)]
    return Pipeline(steps)


def metrics(y_true, predictions):
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def main():
    data_path = locate_training_file()
    df = pd.read_csv(data_path)
    if TARGET not in df:
        raise ValueError("The training CSV must contain a Revenue column.")
    if "ID" in df:
        df = df.drop(columns=["ID"])
    df = df.dropna(subset=[TARGET]).copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(bool)
    if y.nunique() != 2:
        raise ValueError("Revenue needs both True and False examples.")

    needed = ["Administrative", "Informational", "ProductRelated",
              "Administrative_Duration", "Informational_Duration",
              "ProductRelated_Duration", "ExitRates", "BounceRates", "PageValues"]
    missing = [name for name in needed if name not in X]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    raw_columns = X.columns.tolist()
    engineered_columns = add_behavior_features(X.iloc[:1]).columns.tolist()

    # 60% training / 20% threshold-selection / 20% final comparison.
    # The final 20% is untouched until after choosing each model's threshold.
    X_build, X_final, y_build, y_final = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_build, y_build, test_size=0.25, stratify=y_build, random_state=SEED
    )
    print(f"Training data: {data_path}")
    print(f"Rows: training={len(X_train)}, threshold_tuning={len(X_tune)}, final_check={len(X_final)}")
    print(f"Overall purchase rate: {y.mean():.1%}")
    print("The hidden challenge2_test.csv is NOT read or used.\n")

    candidates = [
        "Logistic Regression",
        "HistGradientBoosting",
        "HistGradientBoosting + new features",
        "Random Forest + new features",
    ]
    thresholds = np.round(np.arange(0.15, 0.651, 0.025), 3)
    results = []

    for name in candidates:
        print(f"Testing: {name}")
        model = make_model(name, raw_columns, engineered_columns)
        model.fit(X_train, y_train)
        tune_prob = model.predict_proba(X_tune)[:, 1]
        # Threshold is chosen ONLY using the tuning set, never the final check.
        tune_scores = [f1_score(y_tune, tune_prob >= t, zero_division=0) for t in thresholds]
        best_threshold = float(thresholds[int(np.argmax(tune_scores))])

        final_prob = model.predict_proba(X_final)[:, 1]
        default_results = metrics(y_final, final_prob >= 0.5)
        tuned_predictions = final_prob >= best_threshold
        tuned_results = metrics(y_final, tuned_predictions)
        tn, fp, fn, tp = confusion_matrix(y_final, tuned_predictions, labels=[False, True]).ravel()
        row = {
            "model": name,
            "threshold_from_tuning": best_threshold,
            "tuning_f1": max(tune_scores),
            "final_f1_at_0.50": default_results["f1"],
            "final_precision": tuned_results["precision"],
            "final_recall": tuned_results["recall"],
            "final_f1": tuned_results["f1"],
            "final_average_precision": average_precision_score(y_final, final_prob),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
        results.append(row)
        print(f"  Threshold chosen on tuning set: {best_threshold:.3f}")
        print(f"  Final-check F1 @ 0.50: {default_results['f1']:.4f}")
        print(f"  Final-check F1 @ tuned threshold: {tuned_results['f1']:.4f}\n")

    report = pd.DataFrame(results).sort_values("final_f1", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("=== EXPERIMENT RESULTS (same held-out final check for every model) ===")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    output = Path(__file__).resolve().parent / "model_improvements_results.csv"
    report.to_csv(output, index=False)
    print(f"\nSaved comparison: {output}")
    print("These are experimental holdout results, not official competition scores.")
    print("Do not use the final-check set to repeatedly choose new settings; that would overfit it.")
    print("No changes were made to purchase_prediction.py or predictions.csv.")


if __name__ == "__main__":
    main()
