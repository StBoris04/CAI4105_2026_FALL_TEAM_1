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
display(df_train.drop(columns='ID').describe().T)

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

# Correlation of numeric features with price
num_preview = ['price', 'bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']
display(df_train[num_preview].corr()[['price']].sort_values('price', ascending=False))

# Median price by state (top 10 most expensive states with at least 30 listings)
state_stats = df_train.groupby('state')['price'].agg(['count', 'median'])
display(state_stats[state_stats['count'] >= 30].sort_values('median', ascending=False).head(10))


# PREPARE DATA

def add_features(df):
    """Create engineered features. Uses only values within each row,
    so applying it to train and test separately causes no data leakage."""
    df = df.copy()
    df['sqft_per_bed'] = df['square_feet'] / (df['bedrooms'] + 1)   # +1 avoids divide-by-zero for studios
    df['bath_per_bed'] = df['bathrooms'] / (df['bedrooms'] + 1)
    df['total_rooms']  = df['bedrooms'] + df['bathrooms']
    return df

DROP_COLS = ['ID', 'category']

# Adding training predictors and target
X = add_features(df_train.drop(columns=DROP_COLS + ['price']))
y = df_train['price']

# Test predictors (same transformations, no target available -_-)
X_test = add_features(df_test.drop(columns=DROP_COLS))

# Separate dtypes for preprocessing steps
cat_cols = X.select_dtypes(exclude='number').columns.tolist()   # text columns (works for object or string dtype)
num_cols = [c for c in X.columns if c not in cat_cols]

print('Categorical:', cat_cols)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = 'neg_root_mean_squared_error'   # sklearn maximizes scores, so RMSE is returned as a negative
print('Numeric:    ', num_cols)
print('X shape:', X.shape, '| X_test shape:', X_test.shape)

# Defining processes!
def make_preprocess():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

def make_pipe(model):
    """Wrap any regressor into a full preprocess -> model pipeline."""
    return Pipeline([
        ('preprocess', make_preprocess()),
        ('model', model)
    ])
baseline_models = {
    'Linear Regression':   LinearRegression(),
    'Decision Tree':       DecisionTreeRegressor(random_state=RANDOM_STATE),
    'Random Forest':       RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingRegressor(random_state=RANDOM_STATE),
    'Hist Gradient Boost': HistGradientBoostingRegressor(random_state=RANDOM_STATE),   # extra model
}

def evaluate(name, pipe):
    """Run 5-fold CV and return mean train/validation RMSE and fit time."""
    t0 = time.perf_counter()
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=-1)
    return {
        'Model':          name,
        'Train_RMSE':     -res['train_score'].mean(),
        'CV_RMSE':        -res['test_score'].mean(),
        'CV_RMSE_std':    res['test_score'].std(),
        'Time_sec':       time.perf_counter() - t0,
    }

baseline_results = [evaluate(name, make_pipe(m)) for name, m in baseline_models.items()]
baseline_df = pd.DataFrame(baseline_results).sort_values('CV_RMSE').reset_index(drop=True)
baseline_df['Gap (CV - Train)'] = baseline_df['CV_RMSE'] - baseline_df['Train_RMSE']   # overfitting indicator
baseline_df.round(1)

# Visual comparison of train vs. validation RMSE

ax = baseline_df.set_index('Model')[['Train_RMSE', 'CV_RMSE']].plot(kind='bar', figsize=(8, 4))
ax.set_ylabel('RMSE ($)')
ax.set_title('Baseline models: Train vs. 5-fold CV RMSE')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()

# 7 & 8. Improve the models with Grid Search and select final model

best_name = min(grid_searches, key=lambda n: -grid_searches[n].best_score_)
final_model = grid_searches[best_name].best_estimator_   # already refit on ALL training data (refit=True)

print('Final model:   ', best_name)
print('Best params:   ', grid_searches[best_name].best_params_)
print(f'CV RMSE:        {-grid_searches[best_name].best_score_:.1f}')
print(f'Avg rent:       {y.mean():.1f}  (RMSE as % of mean rent: {-grid_searches[best_name].best_score_ / y.mean():.1%})')

# Feature importance for tree-based final models (helps explain the model) COMMENTED OUT

# model_step = final_model.named_steps['model']
# if hasattr(model_step, 'feature_importances_'):
  #  feat_names = final_model.named_steps['preprocess'].get_feature_names_out()
  #  imp = (pd.Series(model_step.feature_importances_, index=feat_names)
  #           .sort_values(ascending=False).head(15))
 #   imp.sort_values().plot(kind='barh', figsize=(7, 5), title=f'Top 15 feature importances: {best_name}')
 #   plt.tight_layout(); plt.show()
# else:
 #   print(f'{best_name} has no built-in feature_importances_ (use permutation importance during fine-tuning).') ""
test_pred = final_model.predict(X_test)

predictions = pd.DataFrame({'ID': df_test['ID'], 'price': test_pred})
predictions.to_csv('predictions.csv', index=False)

# Sanity checks: one prediction per test row, no missing values, reasonable range
assert len(predictions) == len(df_test)
assert predictions['price'].notna().all()
print('Saved predictions.csv:', predictions.shape)
display(predictions.describe().T)
display(predictions.head())

