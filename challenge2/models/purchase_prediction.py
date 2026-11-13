# 1. Import packages
from IPython.display import display
from pathlib import Path
import time
import platform

# Core data handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Preprocessing + pipelines
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# Validation tools and classification metrics
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate, cross_val_predict,
                                     cross_val_score, GridSearchCV, TunedThresholdClassifierCV)
from sklearn.inspection import permutation_importance

# Required models (+ HistGradientBoosting and a soft-voting ensemble as additional models)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)

# Reproducibility: every random process uses this seed
RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = -1
PROJECT_ROOT = (Path(__file__).resolve().parents[1] if '__file__' in globals()
                else Path.cwd().parent if Path.cwd().name == 'models' else Path.cwd())
np.random.seed(RANDOM_STATE)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
NOTEBOOK_START = time.time()

# 2. Load and inspect
df_train = pd.read_csv(PROJECT_ROOT / 'data' / 'challenge2_train.csv')
df_test = pd.read_csv(PROJECT_ROOT / 'data' / 'challenge2_test.csv')  # final predictions only
TARGET = 'Revenue'
ID_COLUMN = 'ID'

print('Train shape:', df_train.shape, '| Test shape:', df_test.shape)
display(df_train.head())
df_train.info()

# Confirm the two files can use the same feature preparation.
train_predictors = set(df_train.columns) - {TARGET}
test_predictors = set(df_test.columns)
assert train_predictors == test_predictors, 'Train and test predictors must match.'
assert df_train[ID_COLUMN].is_unique and df_test[ID_COLUMN].is_unique, 'IDs must be unique.'
print('Predictor schemas match:', train_predictors == test_predictors)
print('Missing values (train):', df_train.isna().sum().sum())
print('Missing values (test): ', df_test.isna().sum().sum())
print('Duplicate rows (train):', df_train.duplicated().sum())
print('Duplicate rows (test): ', df_test.duplicated().sum())

# Numeric summary, excluding ID because it is an identifier rather than a predictor.
display(df_train.drop(columns=[ID_COLUMN, TARGET]).describe().T)

# The positive class is uncommon, so accuracy alone would be misleading.
class_counts = df_train[TARGET].value_counts().rename_axis(TARGET).to_frame('count')
class_counts['share'] = class_counts['count'] / len(df_train)
display(class_counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
class_counts['count'].plot(kind='bar', ax=axes[0], color=['steelblue', 'darkorange'])
axes[0].set_title('Revenue class counts')
axes[0].set_xlabel('Revenue'); axes[0].set_ylabel('Sessions')
for label, group in df_train.groupby(TARGET):
    axes[1].hist(group['PageValues'], bins=40, alpha=0.55, label=f'Revenue={label}')
axes[1].set_title('PageValues by purchase outcome')
axes[1].set_xlabel('PageValues'); axes[1].set_ylabel('Sessions')
axes[1].legend()
plt.tight_layout()
plt.show()

# Check categorical levels and ensure the test set has no unseen values.
categorical_columns = ['Month', 'OperatingSystems', 'Browser', 'Region',
                       'TrafficType', 'VisitorType', 'Weekend']
for col in categorical_columns:
    unseen = set(df_test[col].dropna()) - set(df_train[col].dropna())
    print(f'{col}: {df_train[col].nunique()} train levels; unseen test levels: {sorted(unseen)}')

# Purchase rate by key categorical variables
month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
rate_month = df_train.groupby('Month')[TARGET].agg(['mean', 'size']).reindex(month_order)
axes[0].bar(rate_month.index, rate_month['mean'], color='steelblue')
axes[0].set_title('Purchase rate by Month'); axes[0].set_ylabel('Share with Revenue=True')
for i, n in enumerate(rate_month['size']):
    axes[0].text(i, rate_month['mean'].iloc[i] + 0.005, f'n={n}', ha='center', fontsize=7)

rate_visitor = df_train.groupby('VisitorType')[TARGET].mean().sort_values()
axes[1].bar(rate_visitor.index, rate_visitor.values, color='darkorange')
axes[1].set_title('Purchase rate by VisitorType')

has_pv = np.where(df_train['PageValues'] > 0, 'PageValues > 0', 'PageValues = 0')
rate_pv = df_train.groupby(has_pv)[TARGET].agg(['mean', 'size'])
axes[2].bar(rate_pv.index, rate_pv['mean'], color='seagreen')
axes[2].set_title('Purchase rate by PageValues > 0')
for i, n in enumerate(rate_pv['size']):
    axes[2].text(i, rate_pv['mean'].iloc[i] + 0.01, f'n={n}', ha='center', fontsize=8)
plt.tight_layout(); plt.show()

display(pd.concat({
    'Weekend': df_train.groupby('Weekend')[TARGET].mean(),
    'SpecialDay>0': df_train.groupby(df_train['SpecialDay'] > 0)[TARGET].mean(),
}).rename('purchase_rate').to_frame().round(3))
display(rate_pv.round(3))

# Skewness of numeric variables and association with the target
numeric_raw = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
               'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
               'PageValues', 'SpecialDay']
assoc = pd.DataFrame({
    'skewness': df_train[numeric_raw].skew(),
    'share_zero': (df_train[numeric_raw] == 0).mean(),
    'spearman_with_Revenue': df_train[numeric_raw].corrwith(df_train[TARGET].astype(int), method='spearman'),
    'mean_if_purchase': df_train.loc[df_train[TARGET], numeric_raw].mean(),
    'mean_if_no_purchase': df_train.loc[~df_train[TARGET], numeric_raw].mean(),
}).sort_values('spearman_with_Revenue', ascending=False)
display(assoc.round(3))

corr = df_train[numeric_raw].corr(method='spearman')
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_raw))); ax.set_xticklabels(numeric_raw, rotation=60, ha='right', fontsize=8)
ax.set_yticks(range(len(numeric_raw))); ax.set_yticklabels(numeric_raw, fontsize=8)
plt.colorbar(im, ax=ax, label='Spearman correlation'); ax.set_title('Correlation among numeric predictors')
plt.tight_layout(); plt.show()

# 3. Prepare predictors and preprocessing. ID is never a model feature.
X = df_train.drop(columns=[ID_COLUMN, TARGET])
y = df_train[TARGET].astype(bool)
X_test = df_test.drop(columns=ID_COLUMN)
numeric_columns = [col for col in X.columns if col not in categorical_columns]
print('Categorical columns:', categorical_columns)
print('Numeric columns:', numeric_columns)

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('categorical', categorical_pipeline, categorical_columns),
    ('numeric', numeric_pipeline, numeric_columns),
])
# 4. Shared validation configuration for every model and Grid Search
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'f1': make_scorer(f1_score, pos_label=True, zero_division=0),
    'precision': make_scorer(precision_score, pos_label=True, zero_division=0),
    'recall': make_scorer(recall_score, pos_label=True, zero_division=0),
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision',
}

# Verify that every fold retains the class balance before modeling.
fold_summary = []
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    fold_summary.append({
        'fold': fold,
        'train_rows': len(train_idx),
        'validation_rows': len(valid_idx),
        'train_positive_rate': y.iloc[train_idx].mean(),
        'validation_positive_rate': y.iloc[valid_idx].mean(),
    })
display(pd.DataFrame(fold_summary).round(4))
print('Primary metric: F1 score for Revenue=True')
print('Supporting metrics: precision, recall, ROC AUC, average precision, and confusion matrix')

# Helper used for every model so that evaluation is identical across experiments.
def evaluate_model(name, estimator, X_data=X, y_data=y, stage=''):
    start = time.time()
    res = cross_validate(estimator, X_data, y_data, cv=cv, scoring=scoring,
                         return_train_score=True, n_jobs=N_JOBS)
    row = {'model': name, 'stage': stage}
    for m in scoring:
        row[f'val_{m}'] = res[f'test_{m}'].mean()
    row['val_f1_std'] = res['test_f1'].std()
    row['train_f1'] = res['train_f1'].mean()
    row['f1_gap'] = row['train_f1'] - row['val_f1']
    row['fit_time_s'] = res['fit_time'].mean()
    row['total_time_s'] = time.time() - start
    return row

summary_cols = ['model', 'stage', 'val_f1', 'val_f1_std', 'val_precision', 'val_recall',
                'val_roc_auc', 'val_average_precision', 'train_f1', 'f1_gap', 'fit_time_s']

# Reference point: a model that never predicts a purchase has F1 = 0 despite ~85% accuracy.
print('Majority-class accuracy:', round(1 - y.mean(), 4), '| Majority-class F1:', 0.0)
