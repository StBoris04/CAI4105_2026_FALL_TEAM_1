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


