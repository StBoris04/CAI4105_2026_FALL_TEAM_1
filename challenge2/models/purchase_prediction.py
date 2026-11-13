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

def required_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

baseline_rows = []
for name, model in required_models().items():
    pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
    baseline_rows.append(evaluate_model(name, pipe, stage='baseline'))
    print(f"{name:24s} val F1 = {baseline_rows[-1]['val_f1']:.4f}")

baseline_df = pd.DataFrame(baseline_rows)[summary_cols].sort_values('val_f1', ascending=False)
display(baseline_df.round(4).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(10, 4))
order = baseline_df.sort_values('val_f1')
ax.barh(order['model'], order['train_f1'], color='lightgray', label='Train F1')
ax.barh(order['model'], order['val_f1'], color='steelblue', label='Validation F1', height=0.5)
ax.set_xlabel('F1 (Revenue=True)'); ax.set_title('Baseline models: train vs validation F1')
ax.legend(loc='lower right'); plt.tight_layout(); plt.show()

LOG_COLUMNS = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
               'ProductRelated', 'ProductRelated_Duration', 'PageValues', 'TotalPages', 'TotalDuration',
               'AvgDurPerPage', 'ProdDurPerPage', 'PageValuePerProduct']

def add_features(df):
    # Row-wise feature engineering. Safe to run inside CV because it learns nothing from data.
    d = df.copy()
    d['TotalPages'] = d['Administrative'] + d['Informational'] + d['ProductRelated']
    d['TotalDuration'] = d['Administrative_Duration'] + d['Informational_Duration'] + d['ProductRelated_Duration']
    d['AvgDurPerPage'] = d['TotalDuration'] / (d['TotalPages'] + 1)
    d['ProdDurPerPage'] = d['ProductRelated_Duration'] / (d['ProductRelated'] + 1)
    d['HasPageValue'] = (d['PageValues'] > 0).astype(int)
    d['PageValuePerProduct'] = d['PageValues'] / (d['ProductRelated'] + 1)
    d['ProductShare'] = d['ProductRelated'] / (d['TotalPages'] + 1)
    d[LOG_COLUMNS] = np.log1p(d[LOG_COLUMNS].clip(lower=0))
    return d

engineered_numeric = [c for c in add_features(X.head()).columns if c not in categorical_columns]
print('Numeric features after engineering:', len(engineered_numeric))
print(engineered_numeric)

fe_preprocessor = ColumnTransformer(transformers=[
    ('categorical', categorical_pipeline, categorical_columns),
    ('numeric', numeric_pipeline, engineered_numeric),
])
def make_pipeline(model):
    # Full pipeline: feature engineering -> impute/encode/scale -> model.
    return Pipeline([
        ('features', FunctionTransformer(add_features, validate=False)),
        ('preprocess', fe_preprocessor),
        ('model', model),
    ])

# Effect of the log transform on skewness
skew_compare = pd.DataFrame({
    'raw_skew': X[numeric_columns].skew(),
    'engineered_skew': add_features(X)[numeric_columns].skew(),
})
display(skew_compare.round(2))

fe_rows = []
for name, model in required_models().items():
    fe_rows.append(evaluate_model(name, make_pipeline(model), stage='feature engineering'))

fe_df = pd.DataFrame(fe_rows)[summary_cols]
fe_compare = (baseline_df[['model', 'val_f1']].rename(columns={'val_f1': 'baseline_val_f1'})
              .merge(fe_df[['model', 'val_f1']].rename(columns={'val_f1': 'fe_val_f1'}), on='model'))
fe_compare['change'] = fe_compare['fe_val_f1'] - fe_compare['baseline_val_f1']
display(fe_compare.sort_values('fe_val_f1', ascending=False).round(4).reset_index(drop=True))


param_grids = {
    'Logistic Regression': (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), {
        'model__C': [0.01, 0.03, 0.1, 0.3, 1, 3],
        'model__class_weight': [None, 'balanced'],
    }),
    'Naive Bayes': (GaussianNB(), {
        'model__var_smoothing': np.logspace(-9, -1, 9),
    }),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {
        'model__n_neighbors': [5, 11, 21, 41],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
    }),
    'Support Vector Machine': (SVC(random_state=RANDOM_STATE), {
        'model__C': [0.3, 1, 3],
        'model__gamma': ['scale', 0.01],
        'model__class_weight': ['balanced'],
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=RANDOM_STATE), {
        'model__max_depth': [3, 5, 7, 10],
        'model__min_samples_leaf': [5, 20, 50],
        'model__class_weight': [None, 'balanced'],
    }),
    'Random Forest': (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1), {
        'model__max_features': ['sqrt', 0.3],
        'model__min_samples_leaf': [1, 3, 5, 10],
        'model__class_weight': [None, 'balanced_subsample'],
    }),
    'Gradient Boosting': (GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE), {
        'model__n_estimators': [150, 300],
        'model__learning_rate': [0.03, 0.1],
        'model__max_depth': [2, 3, 4],
    }),
    'HistGradientBoosting': (HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                                            random_state=RANDOM_STATE), {
        'model__learning_rate': [0.03, 0.1],
        'model__max_leaf_nodes': [15, 31],
        'model__min_samples_leaf': [20, 50],
        'model__class_weight': [None, 'balanced'],
    }),
}

grid_results = {}
for name, (model, grid) in param_grids.items():
    start = time.time()
    search = GridSearchCV(make_pipeline(model), grid, cv=cv, scoring=scoring, refit='f1',
                          return_train_score=True, n_jobs=N_JOBS)
    search.fit(X, y)
    grid_results[name] = search
    best = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    print(f'{name:24s} best CV F1 = {search.best_score_:.4f} | {best} | {time.time() - start:.0f}s')
  # Summarize the best configuration of every grid on all metrics (same folds as the baseline).
tuned_rows = []
for name, search in grid_results.items():
    cvr = search.cv_results_
    i = search.best_index_
    row = {'model': name, 'stage': 'tuned (grid search)'}
    for m in scoring:
        row[f'val_{m}'] = cvr[f'mean_test_{m}'][i]
    row['val_f1_std'] = cvr['std_test_f1'][i]
    row['train_f1'] = cvr['mean_train_f1'][i]
    row['f1_gap'] = row['train_f1'] - row['val_f1']
    row['fit_time_s'] = cvr['mean_fit_time'][i]
    row['best_params'] = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    tuned_rows.append(row)
tuned_df = pd.DataFrame(tuned_rows).sort_values('val_f1', ascending=False)
display(tuned_df[summary_cols + ['best_params']].round(4).reset_index(drop=True))

progress = (baseline_df[['model', 'val_f1']].rename(columns={'val_f1': 'baseline'})
            .merge(fe_compare[['model', 'fe_val_f1']].rename(columns={'fe_val_f1': '+ features'}), on='model', how='outer')
            .merge(tuned_df[['model', 'val_f1']].rename(columns={'val_f1': '+ grid search'}), on='model', how='outer')
            .set_index('model').sort_values('+ grid search'))
ax = progress.plot(kind='barh', figsize=(10, 5), color=['lightgray', 'lightsteelblue', 'steelblue'])
ax.set_xlabel('Validation F1 (Revenue=True)'); ax.set_title('Validation F1 after each improvement step')
plt.tight_layout(); plt.show()

# Top tuned models by validation F1 (tree ensembles have well-behaved probabilities for this step)
top_names = tuned_df.query("model in ['Random Forest', 'Gradient Boosting', 'HistGradientBoosting']")['model'].tolist()
candidates = {name: grid_results[name].best_estimator_ for name in top_names}
candidates['Soft-Voting Ensemble'] = make_pipeline(VotingClassifier(
    estimators=[(n.replace(' ', '_'), grid_results[n].best_estimator_.named_steps['model']) for n in top_names],
    voting='soft'))
# Logistic Regression is kept as an interpretable reference
candidates['Logistic Regression'] = grid_results['Logistic Regression'].best_estimator_

thresholds = np.round(np.arange(0.05, 0.951, 0.005), 3)
oof_proba, threshold_rows = {}, []
for name, est in candidates.items():
    p = cross_val_predict(est, X, y, cv=cv, method='predict_proba', n_jobs=N_JOBS)[:, 1]
    oof_proba[name] = p
    f1s = np.array([f1_score(y, p >= t) for t in thresholds])
    best_t = thresholds[f1s.argmax()]
    threshold_rows.append({
        'model': name,
        'oof_f1_at_0.5': f1_score(y, p >= 0.5),
        'best_threshold': best_t,
        'oof_f1_at_best': f1s.max(),
        'precision_at_best': precision_score(y, p >= best_t),
        'recall_at_best': recall_score(y, p >= best_t),
        'roc_auc': roc_auc_score(y, p),
        'average_precision': average_precision_score(y, p),
    })
threshold_df = pd.DataFrame(threshold_rows).sort_values('oof_f1_at_best', ascending=False)
display(threshold_df.round(4).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(10, 4))
for name, p in oof_proba.items():
    ax.plot(thresholds, [f1_score(y, p >= t) for t in thresholds], label=name)
ax.axvline(0.5, color='gray', ls='--', lw=1)
ax.set_xlabel('Probability threshold'); ax.set_ylabel('Out-of-fold F1'); ax.set_ylim(0.3, 0.75)
ax.set_title('F1 vs decision threshold (out-of-fold)'); ax.legend(); plt.tight_layout(); plt.show()

inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1)

def with_tuned_threshold(estimator):
    return TunedThresholdClassifierCV(estimator, scoring='f1', cv=inner_cv,
                                      thresholds=200, random_state=RANDOM_STATE)

nested_rows = []
for name in threshold_df['model'].head(4):
    start = time.time()
    scores = cross_val_score(with_tuned_threshold(candidates[name]), X, y, cv=cv,
                             scoring=scoring['f1'], n_jobs=N_JOBS)
    nested_rows.append({'model': name, 'nested_f1_mean': scores.mean(), 'nested_f1_std': scores.std(),
                        'fold_scores': np.round(scores, 4), 'time_s': round(time.time() - start)})
    print(f'{name:22s} nested F1 = {scores.mean():.4f} +/- {scores.std():.4f}')
nested_df = pd.DataFrame(nested_rows).sort_values('nested_f1_mean', ascending=False).reset_index(drop=True)
display(nested_df.round(4))

#5.2 Final comparison table

final_table = (tuned_df[['model', 'val_f1', 'train_f1', 'f1_gap', 'val_roc_auc', 'val_average_precision', 'fit_time_s']]
               .rename(columns={'val_f1': 'tuned_f1@0.5'})
               .merge(threshold_df[['model', 'best_threshold', 'oof_f1_at_best']], on='model', how='outer')
               .merge(nested_df[['model', 'nested_f1_mean', 'nested_f1_std']], on='model', how='left')
               .sort_values(['nested_f1_mean', 'oof_f1_at_best', 'tuned_f1@0.5'], ascending=False, na_position='last')
               .reset_index(drop=True))
display(final_table.round(4))

FINAL_MODEL_NAME = nested_df.iloc[0]['model']
# Prefer a single model over the ensemble when the ensemble's advantage is within noise
single = nested_df[nested_df['model'] != 'Soft-Voting Ensemble'].iloc[0]
if FINAL_MODEL_NAME == 'Soft-Voting Ensemble':
    gap = nested_df.iloc[0]['nested_f1_mean'] - single['nested_f1_mean']
    if gap < nested_df.iloc[0]['nested_f1_std'] / 2:
        FINAL_MODEL_NAME = single['model']
print('Selected final model:', FINAL_MODEL_NAME)

p_final = oof_proba[FINAL_MODEL_NAME]
t_final_oof = threshold_df.set_index('model').loc[FINAL_MODEL_NAME, 'best_threshold']
pred_default = p_final >= 0.5
pred_tuned = p_final >= t_final_oof

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pred, title in [(axes[0], pred_default, 'threshold 0.50'), (axes[1], pred_tuned, f'threshold {t_final_oof:.3f}')]:
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=['False', 'True']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{FINAL_MODEL_NAME}\nout-of-fold, {title} (F1={f1_score(y, pred):.3f})')
plt.tight_layout(); plt.show()

print('Note: threshold was chosen using these same out-of-fold targets; the report below is exploratory, not an independent score.')
print(f'Classification report, out-of-fold, threshold {t_final_oof:.3f}:')
print(classification_report(y, pred_tuned, target_names=['No purchase', 'Purchase'], digits=4))

# Fit on one CV training split and measure the drop in average precision on its validation split.
train_idx, valid_idx = next(cv.split(X, y))
imp_model = candidates[FINAL_MODEL_NAME]
imp_model = sklearn.base.clone(imp_model).fit(X.iloc[train_idx], y.iloc[train_idx])
perm = permutation_importance(imp_model, X.iloc[valid_idx], y.iloc[valid_idx], scoring='average_precision',
                              n_repeats=10, random_state=RANDOM_STATE, n_jobs=N_JOBS)
importance = (pd.DataFrame({'feature': X.columns, 'importance_mean': perm.importances_mean,
                            'importance_std': perm.importances_std})
              .sort_values('importance_mean', ascending=True))
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance['feature'], importance['importance_mean'], xerr=importance['importance_std'], color='steelblue')
ax.set_xlabel('Drop in average precision when shuffled'); ax.set_title(f'Permutation importance: {FINAL_MODEL_NAME}')
plt.tight_layout(); plt.show()

final_model = with_tuned_threshold(candidates[FINAL_MODEL_NAME])
final_model.fit(X, y)
print('Final model:', FINAL_MODEL_NAME)
print('Hyperparameters:', {k.replace('model__', ''): v for k, v in grid_results[FINAL_MODEL_NAME].best_params_.items()}
      if FINAL_MODEL_NAME in grid_results else 'ensemble of tuned models')
print(f'Decision threshold chosen on training data: {final_model.best_threshold_:.4f}')
print(f'Inner-CV F1 at that threshold: {final_model.best_score_:.4f}')

test_pred = final_model.predict(X_test).astype(bool)
predictions = pd.DataFrame({ID_COLUMN: df_test[ID_COLUMN], TARGET: test_pred})

# Submission checks
assert len(df_test) > 0, 'Test dataset is empty.'
assert len(predictions) == len(df_test)
assert predictions[ID_COLUMN].is_unique and predictions[ID_COLUMN].equals(df_test[ID_COLUMN])
assert predictions[TARGET].isin([True, False]).all() and predictions[TARGET].notna().all()

out_path = PROJECT_ROOT / 'submissions' / 'predictions.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(out_path, index=False)
print('Saved', out_path, predictions.shape)
print(f'Predicted purchase rate on test: {predictions[TARGET].mean():.3f} '
      f'(training purchase rate {y.mean():.3f}; OOF predicted rate {pred_tuned.mean():.3f})')
display(predictions.head())
display(predictions[TARGET].value_counts())

print('Python      ', platform.python_version())
for pkg in [np, pd, sklearn, plt.matplotlib]:
    print(f'{pkg.__name__:12s}', pkg.__version__)
print(f'Total notebook runtime: {(time.time() - NOTEBOOK_START) / 60:.1f} minutes')
