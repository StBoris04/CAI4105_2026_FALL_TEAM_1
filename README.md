# CAI4105 Team 1

This project develops a reproducible machine learning workflow to predict monthly
apartment rental prices from listing characteristics and location. The supplied
data has already received basic cleaning, so the focus is model development,
validation, comparison, and improvement.

## Data

- `datasets/challenge1_train.csv`: 7,398 labeled observations. The target is
  `price`.
- `datasets/challenge1_test.csv`: 2,467 observations without prices, reserved
  for final prediction generation.

The `ID` column identifies each listing and must not be used as a predictor. The
test set must not be used for model development, validation, feature engineering,
or tuning. The original dataset is from the [UCI Apartment for Rent Classified
dataset](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified).

## Modeling approach

The project evaluates Linear Regression, Decision Tree, Random Forest, and
Gradient Boosting regression models. Model selection and tuning use training data
only, with RMSE as the primary metric. The final choice should balance validation
performance with generalization, complexity, computation, and interpretability.

The current notebook comparison uses shuffled 5-fold cross-validation with
`random_state=42` and RMSE. It reports the following results:

| Model | Baseline CV RMSE | Tuned CV RMSE |
|---|---:|---:|
| Linear Regression | 526.82 | 526.82 |
| Decision Tree | 555.96 | 493.66 |
| Random Forest | 426.71 | 418.65 |
| Gradient Boosting | 461.27 | **417.08** |

Gradient Boosting is the current selected model, using 200 estimators, depth 4,
learning rate 0.1, and subsample 1.0. Its reported tuned score is the mean RMSE
from the same 5-fold validation used for final comparison. Random Forest is close
at 418.65, so the small score difference should be treated as a near tie rather
than a decisive performance gap. These results are those recorded in the current
notebook and should be regenerated when the workflow is finalized.

## Project layout

- `models/`: reusable model implementation.
- `datasets/`: supplied training and test datasets.
- `Rental_Price_Prediction.ipynb`: complete exploration, validation, model
  comparison, tuning, and prediction workflow (add the notebook to this repo
  before final submission).

## Reproduce predictions

The baseline pipeline excludes `ID` and the nearly constant `category` feature,
standardizes numeric features, and one-hot encodes `has_photo` and `state`,
ignoring categories not seen during training.

Install Python with pandas and scikit-learn, then run from the repository root:

```bash
python models/base_line_model.py
```

The script writes `artifacts/submission.csv` with `ID` and predicted `price`.
The assignment requires a file named `predictions.csv`; rename or configure the
output accordingly for the final submission. The final GitHub submission should
include the completed notebook, README, predictions, runnable code, and
environment/dependency information.
