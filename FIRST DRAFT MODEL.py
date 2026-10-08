# First draft for apartment predictor

# 1. Import packages

# Core data handling
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# Preprocessing + pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Validation tools
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)

# Reproducibility: every random process uses this seed
RANDOM_STATE = 42

# 2. Load and inspect

# Load both datasets
df_train = pd.read_csv('challenge1_train.csv')
df_test  = pd.read_csv('challenge1_test.csv')   # used ONLY for final predictions

print('Train shape:', df_train.shape, '| Test shape:', df_test.shape)
display(df_train.head())
df_train.info()

# Quick peek at the numeric stats (skipping ID since it's just a useless tag)
display(df_train.drop(columns='ID').describe().T)[cite: 1, 2]

# Supposedly the data is clean, but let's double-check that no missing values snuck in[cite: 1, 2]
print('Missing values (train):', df_train.isna().sum().sum())[cite: 1, 2]
print('Missing values (test): ', df_test.isna().sum().sum())[cite: 1, 2]

# Let's see what's going on with the text categories[cite: 1, 2]
for col in ['category', 'has_photo', 'state']:[cite: 1, 2]
    print(f"\n{col}: {df_train[col].nunique()} unique values")[cite: 1, 2]
    print(df_train[col].value_counts().head(10))[cite: 1, 2]

# Catching any sneaky states in the test data that didn't show up during training[cite: 1, 2]
unseen_states = set(df_test['state']) - set(df_train['state'])[cite: 1, 2]
print('\nStates in test but not in train:', unseen_states)[cite: 1, 2]
