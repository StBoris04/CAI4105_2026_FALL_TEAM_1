"""Archived baseline scratch check.

The canonical Challenge 2 workflow is ``purchase_prediction.ipynb``. Do not
use this file for model selection or submission: it uses a separate holdout
split and reads the test CSV before Task 6, which conflicts with the project
validation rules. It is retained only as historical reference.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
train_path = ROOT / 'data' / 'challenge2_train.csv'
test_path = ROOT / 'data' / 'challenge2_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Validate input files before training.
assert len(train) == 9247 and len(test) == 3083, 'Unexpected dataset size'
assert 'Revenue' in train.columns and 'Revenue' not in test.columns
assert set(train.columns) - {'Revenue'} == set(test.columns), 'Feature columns differ'
assert train['ID'].is_unique and test['ID'].is_unique, 'Duplicate ID found'
assert not train['ID'].isin(test['ID']).any(), 'An ID occurs in both files'
assert not train.isna().any().any() and not test.isna().any().any(), 'Missing values found'

print(f'Train: {train.shape}; test: {test.shape}')
print('Training Revenue counts:', train['Revenue'].value_counts().to_dict())
print(f'Positive share: {train["Revenue"].mean():.1%}')

# ID uniquely identifies a session; it is NOT a predictor.
X = train.drop(columns=['Revenue', 'ID'])
y = train['Revenue'].astype(bool)

# Stratification preserves approximately the same buyer/nonbuyer proportions.
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

categories = ['Month', 'OperatingSystems', 'Browser', 'Region',
              'TrafficType', 'VisitorType', 'Weekend']
numerics = [c for c in X.columns if c not in categories]

# All learned preprocessing stays INSIDE the pipeline to prevent data leakage.
preprocess = ColumnTransformer([
    ('categorical', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore')),
    ]), categories),
    ('numeric', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), numerics),
])

model = Pipeline([
    ('preprocess', preprocess),
    ('model', LogisticRegression(max_iter=5000, random_state=42)),
])
model.fit(X_tr, y_tr)
pred = model.predict(X_val)

print(f'Holdout rows: training={len(X_tr)}, validation={len(X_val)}')
print(f'Precision (Revenue=True): {precision_score(y_val, pred, zero_division=0):.4f}')
print(f'Recall    (Revenue=True): {recall_score(y_val, pred, zero_division=0):.4f}')
print(f'F1        (Revenue=True): {f1_score(y_val, pred, zero_division=0):.4f}')
print('Confusion matrix [[TN, FP], [FN, TP]]:')
print(confusion_matrix(y_val, pred, labels=[False, True]))
print('Baseline complete. The hidden test dataset was not used for evaluation.')
