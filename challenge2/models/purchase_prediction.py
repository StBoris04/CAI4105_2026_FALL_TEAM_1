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
