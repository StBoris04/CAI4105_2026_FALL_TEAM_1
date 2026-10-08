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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
import xgboost

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

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Distribution of price
axes[0].hist(df_train['price'], bins=60)
axes[0].set_title('Price distribution')
axes[0].set_xlabel('Monthly rent ($)')

# 2. Price vs. square feet (size should be a strong driver)
axes[1].scatter(df_train['square_feet'], df_train['price'], s=4, alpha=0.3)
axes[1].set_title('Price vs. Square Feet')
axes[1].set_xlabel('Square feet'); axes[1].set_ylabel('Price ($)')

# 3. Location map colored by price (location should also be a strong driver)
sc = axes[2].scatter(df_train['longitude'], df_train['latitude'],
                     c=np.clip(df_train['price'], 0, 4000), s=4, cmap='viridis')
axes[2].set_title('Location colored by price (capped at $4k)')
axes[2].set_xlabel('Longitude'); axes[2].set_ylabel('Latitude')
plt.colorbar(sc, ax=axes[2])

plt.tight_layout()
plt.show()
