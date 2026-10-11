# CAI4105 Team 1 — Apartment Rental Price Prediction

Predict monthly apartment rent from listing characteristics and location.

## Files

- `datasets/challenge1_train.csv`: 7,398 labeled listings with target `price`.
- `datasets/challenge1_test.csv`: 2,467 listings used only for final predictions.
- `FIRST DRAFT MODEL.py`: the complete modeling workflow.
- `models/rent_prediction.ipynb`: the same workflow as a Jupyter notebook.
- `predictions.csv`: the generated submission with columns `ID` and `price`.

`ID` is retained for the submission and never used as a predictor.

## Run

Install Python 3.11 or newer and the required packages:

```bash
python -m pip install numpy pandas matplotlib scikit-learn xgboost notebook ipykernel
```

From any directory, run the script by path, or open the notebook and run all cells:

```bash
python "FIRST DRAFT MODEL.py"
jupyter notebook models/rent_prediction.ipynb
```

Both versions read the CSVs from `datasets/` and write `predictions.csv` to the repository root. Grid Search can take several minutes.

## Method

The training data is explored and prepared with apartment-layout features, one-hot encoding of categories, and numeric scaling. The same shuffled five folds and dollar RMSE are used to compare Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and optional SGD, Histogram Gradient Boosting, and XGBoost models. Selected models are tuned with Grid Search. Preprocessing is fit inside each fold's pipeline.

The final model is selected from both baseline and tuned candidates using validation RMSE, then fit on all labeled rows. The workflow prints a comparison table and a justification covering validation performance, overfitting, complexity, computation, and interpretability. It verifies the saved prediction file against every test ID. The selected CV score may be optimistic because the same data guided model selection; hidden test prices are never used.
