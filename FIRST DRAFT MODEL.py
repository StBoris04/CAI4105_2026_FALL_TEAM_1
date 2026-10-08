# First draft for apartment predictor

# 1. Import packages

# Core data handling
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# Preprocessing + pipelines
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Validation tools
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Regression models (4 required + 1 extra)
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
