"""Challenge 2: compare team feature recipe vs new feature recipe fairly.

Place beside model_improvements.py in challenge2/models/.
Run from repository outer directory using the Python 3.11 .venv311 interpreter.
Does not import, execute, or change purchase_prediction.py.
Does not read challenge2_test.csv or write predictions.csv.

IMPORTANT: This compares FEATURE RECIPES using matched learners, not the team's
fully tuned final system. The seed=42 holdout was previously inspected, so these
are exploratory measurements rather than a fresh independent test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
TARGET = 'Revenue'
CATEGORIES = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
TEAM_LOG = ['Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'PageValues', 'TotalPages', 'TotalDuration', 'AvgDurPerPage',
            'ProdDurPerPage', 'PageValuePerProduct']


def locate_data():
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / 'data' / 'challenge2_train.csv',
                          parent / 'challenge2' / 'data' / 'challenge2_train.csv'):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError('Put this file in challenge2/models/ with challenge2/data/challenge2_train.csv')


def team_features(df):
    # Reproduces the team's add_features() transformation in purchase_prediction.py.
    d = df.copy()
    d['TotalPages'] = d['Administrative'] + d['Informational'] + d['ProductRelated']
    d['TotalDuration'] = d['Administrative_Duration'] + d['Informational_Duration'] + d['ProductRelated_Duration']
    d['AvgDurPerPage'] = d['TotalDuration'] / (d['TotalPages'] + 1)
    d['ProdDurPerPage'] = d['ProductRelated_Duration'] / (d['ProductRelated'] + 1)
    d['HasPageValue'] = (d['PageValues'] > 0).astype(int)
    d['PageValuePerProduct'] = d['PageValues'] / (d['ProductRelated'] + 1)
    d['ProductShare'] = d['ProductRelated'] / (d['TotalPages'] + 1)
    d[TEAM_LOG] = np.log1p(d[TEAM_LOG].clip(lower=0))
    return d


def new_features(df):
    # Same recipe as model_improvements.py.
    d = df.copy()
    pages = d['Administrative'] + d['Informational'] + d['ProductRelated']
    duration = d['Administrative_Duration'] + d['Informational_Duration'] + d['ProductRelated_Duration']
    d['TotalPages'] = pages
    d['TotalDuration'] = duration
    d['AvgDurationPerPage'] = duration / (pages + 1)
    d['ProductPageShare'] = d['ProductRelated'] / (pages + 1)
    d['ProductDurationShare'] = d['ProductRelated_Duration'] / (duration + 1)
    d['ExitMinusBounce'] = d['ExitRates'] - d['BounceRates']
    d['PageValuePerProductPage'] = d['PageValues'] / (d['ProductRelated'] + 1)
    d['ProductDurationPerPage'] = d['ProductRelated_Duration'] / (d['ProductRelated'] + 1)
    d['HasPageValue'] = (d['PageValues'] > 0).astype(int)
    return d


def pipeline(learner_name, recipe_name, X):
    feature_fn = team_features if recipe_name == 'Team features' else new_features
    out_columns = feature_fn(X.iloc[:1]).columns
    categorical = [c for c in CATEGORIES if c in out_columns]
    numeric = [c for c in out_columns if c not in categorical]
    prep = ColumnTransformer([
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric),
    ])
    if learner_name == 'Random Forest':
        # Same learner settings as previous experiment for matched comparison.
        learner = RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                         random_state=SEED, n_jobs=-1)
    else:
        learner = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                                 max_leaf_nodes=15, l2_regularization=1.0,
                                                 random_state=SEED)
    return Pipeline([
        ('features', FunctionTransformer(feature_fn, validate=False)),
        ('preprocess', prep),
        ('model', learner),
    ])


def main():
    path = locate_data()
    df = pd.read_csv(path).dropna(subset=[TARGET])
    X = df.drop(columns=['ID', TARGET], errors='ignore')
    y = df[TARGET].astype(bool)
    # Match the earlier model_improvements.py split exactly.
    X_build, X_check, y_build, y_check = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_fit, X_tune, y_fit, y_tune = train_test_split(
        X_build, y_build, test_size=0.25, stratify=y_build, random_state=SEED)
    thresholds = np.round(np.arange(0.15, 0.651, 0.025), 3)
    print('Comparing TEAM features against NEW features on identical rows and learners.')
    print('Training:', len(X_fit), 'Threshold tuning:', len(X_tune), 'Final check:', len(X_check))
    print('NOTE: final-check rows were already examined in the previous experiment.')
    print('Therefore this is exploratory, not an untouched independent test.')
    print('The team\'s full grid search and ensemble are NOT being evaluated here.')
    print('Hidden challenge2_test.csv is NOT used.\n')
    results = []
    for learner_name in ['Random Forest', 'HistGradientBoosting']:
        for recipe_name in ['Team features', 'New features']:
            model = pipeline(learner_name, recipe_name, X)
            model.fit(X_fit, y_fit)
            tune_prob = model.predict_proba(X_tune)[:, 1]
            tune_f1 = [f1_score(y_tune, tune_prob >= t, zero_division=0) for t in thresholds]
            threshold = float(thresholds[int(np.argmax(tune_f1))])
            final_prob = model.predict_proba(X_check)[:, 1]
            pred = final_prob >= threshold
            row = dict(model=learner_name, features=recipe_name, threshold=threshold,
                       tuning_f1=max(tune_f1),
                       check_f1=f1_score(y_check, pred, zero_division=0),
                       check_precision=precision_score(y_check, pred, zero_division=0),
                       check_recall=recall_score(y_check, pred, zero_division=0))
            results.append(row)
            print(f"{learner_name:20s} | {recipe_name:13s} | F1={row['check_f1']:.4f} | Precision={row['check_precision']:.4f} | Recall={row['check_recall']:.4f} | threshold={threshold:.3f}")
    report = pd.DataFrame(results)
    print('\nDifference (new features minus team features; positive means new scored higher):')
    for learner_name, group in report.groupby('model', sort=False):
        team = group.loc[group.features == 'Team features', 'check_f1'].iloc[0]
        new = group.loc[group.features == 'New features', 'check_f1'].iloc[0]
        print(f'  {learner_name}: {new - team:+.4f} F1')
    output = Path(__file__).resolve().parent / 'team_feature_comparison.csv'
    report.to_csv(output, index=False)
    print('\nSaved:', output)
    print('Do not edit the team model or claim an improvement based only on this comparison.')


if __name__ == '__main__':
    main()
