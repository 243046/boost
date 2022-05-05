import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_classification


class LightGBMExperiment:
    def __init__(self):
        self.model = LGBMClassifier(n_estimators=150, boosting_type='goss')
        self.param_grid = {
        'learning_rate': np.linspace(0.03, 0.3, 10),
        'num_leaves': np.linspace(13, 130, 10).astype(int),
        'top_rate': np.linspace(0.1, 0.5, 10),
        'other_rate': np.linspace(0.05, 0.2, 10),
        'reg_alpha': np.linspace(0, 1, 10),
        'reg_lambda': np.linspace(0, 3, 10)
        }

    def _make_grid(self):
        return [{param: values_list} for param, values_list in self.param_grid.items()]

    def fit(self):
        X, y = make_classification(n_samples=10000, n_features=20, class_sep=0.3, random_state=123)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
        grid = self._make_grid()
        model = GridSearchCV(self.model, param_grid=grid, cv=cv, scoring='accuracy', n_jobs=-1)
        model.fit(X, y)
        self.cv_results_ = model.cv_results_
        return self