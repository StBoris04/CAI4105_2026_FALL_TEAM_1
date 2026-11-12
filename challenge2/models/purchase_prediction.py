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
