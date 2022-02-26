import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def sigmoid(x):
    return 1/(1+np.exp(-x))


def calculate_importances(models, n_features=5):
    np.random.seed(123)
    X = np.random.normal(size=(1000, n_features))
    features = [f'X_{i}' for i in range(n_features)]
    X = pd.DataFrame(X, columns=features)
    beta = 3**(np.arange(X.shape[1], 0, -1) -1)
    print(f'weights = {list(beta)}')
    y = 1*(sigmoid(X @ beta) > .5)

    importances = [model.fit(X, y).feature_importances_ for model in models.values()]
    importances = [x/x.max() for x in importances]
    importances = pd.DataFrame(importances, columns=features)
    importances['model'] = models.keys()
    return importances


if __name__ == '__main__':
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(n_estimators=100, verbose=False, random_state=123)
    }

    importances = calculate_importances(models)
    importances.to_excel('../feature_importance/importances/importances.xlsx', index=False)
