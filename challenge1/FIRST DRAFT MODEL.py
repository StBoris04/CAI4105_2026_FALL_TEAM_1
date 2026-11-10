# First draft for apartment predictor

# 1. Import packages

# Core data handling
from pathlib import Path
from IPython.display import display
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
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)

# Reproducibility: every random process uses this seed
RANDOM_STATE = 42
# Support the root script and its notebook copy in models/.
PROJECT_ROOT = (Path(__file__).resolve().parent if '__file__' in globals()
                else Path.cwd().parent if Path.cwd().name == 'models' else Path.cwd())

# 2. Load and inspect

# Load both datasets
df_train = pd.read_csv(PROJECT_ROOT / 'datasets' / 'challenge1_train.csv')
df_test  = pd.read_csv(PROJECT_ROOT / 'datasets' / 'challenge1_test.csv')   # used ONLY for final predictions

print('Train shape:', df_train.shape, '| Test shape:', df_test.shape)
display(df_train.head())
df_train.info()

# Quick peek at the numeric stats (skipping ID since it's just a useless tag)
display(df_train.drop(columns='ID').describe().T)

# Supposedly the data is clean, but let's double-check that no missing values snuck in
print('Missing values (train):', df_train.isna().sum().sum())
print('Missing values (test): ', df_test.isna().sum().sum())

# Let's see what's going on with the text categories
for col in ['category', 'has_photo', 'state']:
    print(f"\n{col}: {df_train[col].nunique()} unique values")
    print(df_train[col].value_counts().head(10))

# Catching any sneaky states in the test data that didn't show up during training
unseen_states = set(df_test['state']) - set(df_train['state'])
print('\nStates in test but not in train:', unseen_states)

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
    'Random Forest':       RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
    'Gradient Boosting':   GradientBoostingRegressor(random_state=RANDOM_STATE),
    'Hist Gradient Boost': HistGradientBoostingRegressor(random_state=RANDOM_STATE),   # extra model
}

def evaluate(name, pipe):
    """Run 5-fold CV and return mean train/validation RMSE and fit time."""
    t0 = time.perf_counter()
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=4)
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
display(baseline_df.round(1))

# Visual comparison of train vs. validation RMSE

ax = baseline_df.set_index('Model')[['Train_RMSE', 'CV_RMSE']].plot(kind='bar', figsize=(8, 4))
ax.set_ylabel('RMSE ($)')
ax.set_title('Baseline models: Train vs. 5-fold CV RMSE')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()

# 7. Improve selected models with Grid Search
# Reuse the same shuffled five folds and dollar RMSE for every comparison.
# Pipelines fit encoding/scaling only on each fold's training rows.
# These tuning scores guide selection; they are not an independent test estimate.
param_grids = {
    'Decision Tree': (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {'model__max_depth': [8, 12, 16, None],
         'model__min_samples_leaf': [1, 5, 10]},
    ),
    'Random Forest': (
        RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        {'model__max_features': [0.5, 1.0], 'model__min_samples_leaf': [1, 3]},
    ),
    'Gradient Boosting': (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {'model__n_estimators': [300], 'model__learning_rate': [0.05, 0.1],
         'model__max_depth': [3, 5], 'model__subsample': [0.8, 1.0]},
    ),
    'Hist Gradient Boost': (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {'model__learning_rate': [0.05, 0.1], 'model__max_iter': [300, 600],
         'model__max_leaf_nodes': [31, 63]},
    ),
}

grid_searches = {}
tuned_rows = []
for name, (model, grid) in param_grids.items():
    start = time.perf_counter()
    search = GridSearchCV(
        make_pipe(model), grid, cv=cv, scoring=SCORING,
        return_train_score=True, n_jobs=4, refit=True, error_score='raise'
    )
    search.fit(X, y)
    grid_searches[name] = search
    best = search.best_index_
    tuned_rows.append({
        'Model': name + ' (tuned)',
        'Train_RMSE': -search.cv_results_['mean_train_score'][best],
        'CV_RMSE': -search.best_score_,
        'CV_RMSE_std': search.cv_results_['std_test_score'][best],
        'Time_sec': time.perf_counter() - start,
    })
    print(name, 'best parameters:', search.best_params_, flush=True)

# 8. Compare all models and select the final model
comparison_df = pd.concat([
    baseline_df.drop(columns='Gap (CV - Train)'), pd.DataFrame(tuned_rows)
], ignore_index=True).sort_values('CV_RMSE').reset_index(drop=True)
comparison_df['Gap (CV - Train)'] = comparison_df['CV_RMSE'] - comparison_df['Train_RMSE']
display(comparison_df.round(2))

# Include the baseline candidates: tuning is not guaranteed to improve them.
candidate_models = {name: make_pipe(model) for name, model in baseline_models.items()}
candidate_models.update({name + ' (tuned)': search.best_estimator_
                         for name, search in grid_searches.items()})
best_row = comparison_df.iloc[0]
best_name = best_row['Model']
final_model = candidate_models[best_name]
start = time.perf_counter()
final_model.fit(X, y)  # Fit the selected pipeline on every labeled row.
final_fit_seconds = time.perf_counter() - start

# Explain the actual selection, including all five assignment criteria.
model_step = final_model.named_steps['model']
params = model_step.get_params()
if isinstance(model_step, (LinearRegression, SGDRegressor)):
    complexity = 'a linear model with one coefficient per transformed feature'
    interpretability = 'Its coefficients are inspectable, although correlated features complicate interpretation.'
elif isinstance(model_step, DecisionTreeRegressor):
    complexity = f'a single tree of depth {model_step.get_depth()} with {model_step.get_n_leaves()} leaves'
    interpretability = 'Its decision paths can be inspected, but large trees are difficult to summarize.'
elif isinstance(model_step, HistGradientBoostingRegressor):
    complexity = f"a boosting ensemble with {model_step.n_iter_} iterations and max_leaf_nodes={params['max_leaf_nodes']}"
    interpretability = 'It is less directly interpretable than a linear model or one tree; permutation importance can summarize feature influence.'
else:
    complexity = f"an ensemble of {params['n_estimators']} trees with max_depth={params['max_depth']}"
    interpretability = 'It is less directly interpretable than a linear model or one tree; feature importances can summarize influence but do not establish causation.'

print(f"Selected model: {best_name}. Its mean validation RMSE was ${best_row['CV_RMSE']:.2f} "
      f"(fold standard deviation ${best_row['CV_RMSE_std']:.2f}), the lowest among all candidates. "
      f"Training RMSE was ${best_row['Train_RMSE']:.2f}, giving a train-validation gap of "
      f"${best_row['Gap (CV - Train)']:.2f}; a positive gap indicates remaining overfitting. "
      f"The selected estimator is {complexity}. Its evaluation/search took "
      f"{best_row['Time_sec']:.1f}s, while the final full-data fit took {final_fit_seconds:.1f}s. "
      f"We accept this computational cost to prioritize validation RMSE. {interpretability} "
      "The CV scores guided hyperparameter and model selection, so the winning score may be optimistic. "
      "Small differences between candidates do not establish statistical significance; "
      "performance on the hidden test prices remains unknown.")

# 9. Generate the required predictions
predictions = pd.DataFrame({'ID': df_test['ID'], 'price': final_model.predict(X_test)})
assert list(predictions.columns) == ['ID', 'price']
assert len(predictions) == len(df_test) == 2467
assert predictions['ID'].is_unique
assert predictions['ID'].equals(df_test['ID'])
assert np.isfinite(predictions['price']).all()
predictions.to_csv(PROJECT_ROOT / 'predictions.csv', index=False)
print('Saved predictions.csv:', predictions.shape)
display(predictions.head())
display(predictions['price'].describe())
