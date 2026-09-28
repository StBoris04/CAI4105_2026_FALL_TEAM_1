"""Baseline apartment rent model based on Rental_Price_Prediction.ipynb."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
NUMERIC_FEATURES = ["bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]
CATEGORICAL_FEATURES = ["has_photo", "state"]
DROP_FEATURES = ["ID", "category"]


def build_model() -> Pipeline:
    """Build the selected Gradient Boosting pipeline with notebook parameters."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    regressor = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        subsample=1.0,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
    )
    return Pipeline([("prep", preprocessor), ("model", regressor)])


def load_data(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the supplied train and test CSVs from the repository's datasets/ folder."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "datasets"
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "challenge1_train.csv")
    test = pd.read_csv(data_dir / "challenge1_test.csv")
    return train, test


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate features and target, excluding identifiers and the weak category field."""
    target = train["price"]
    train_features = train.drop(columns=DROP_FEATURES + ["price"])
    test_features = test.drop(columns=DROP_FEATURES)
    return train_features, target, test_features


def train_and_predict(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit on all labeled rows and return nonnegative rent predictions keyed by ID."""
    features, target, test_features = prepare_features(train, test)
    model = build_model()
    model.fit(features, target)
    predictions = model.predict(test_features).clip(min=0)
    submission = pd.DataFrame({"ID": test["ID"], "price": predictions.round(2)})
    return model, submission


def main() -> None:
    train, test = load_data()
    _, submission = train_and_predict(train, test)
    output_path = Path(__file__).resolve().parents[1] / "artifacts" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    main()
