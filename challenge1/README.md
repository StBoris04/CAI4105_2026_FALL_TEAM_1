# Challenge 1: Apartment rent prediction

This folder contains the existing Challenge 1 work, isolated from Challenge 2.

## Files

- `FIRST DRAFT MODEL.py`: modeling, comparison, tuning, final selection, and prediction generation.
- `models/rent_prediction.ipynb`: notebook version of the workflow.
- `datasets/`: supplied Challenge 1 training and test data.
- `predictions.csv`: existing generated submission.
- `datasets.zip`: original dataset archive.

The training set target is `price`; `ID` is not used as a predictor. The workflow
uses shuffled five-fold cross-validation and RMSE. From this directory, run
`python "FIRST DRAFT MODEL.py"` or open `models/rent_prediction.ipynb`.

Dependencies: Python 3.11+, numpy, pandas, matplotlib, scikit-learn, xgboost,
notebook, and ipykernel. The script writes its output to this directory's
`predictions.csv`.
