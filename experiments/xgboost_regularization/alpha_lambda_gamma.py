import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from data_processing.process_dataset import prepare_datasets_for_classification


class Regularization:
    def __init__(self, model, param_grid, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring

    def fit(self, X, y):
        X_train, X_test, _ , _ = train_test_split(X, y, test_size=.3, stratify=y, random_state=123)
        cv = [(X_train.index, X_test.index)]
        search = GridSearchCV(self.model,
                             param_grid=self.param_grid,
                             scoring=self.scoring,
                             cv=cv,
                             n_jobs=-1
                             )
        search.fit(X, y)
        self.search_results_ = search.cv_results_

    def get_search_results(self):
        return (
            pd.DataFrame(self.search_results_)
            .drop(columns='params')
            .filter(regex=r'(param)|(mean_test_score)', axis=1)
            .rename(columns=lambda x: x.removeprefix('param_'))
            .rename(columns={'mean_test_score': 'accuracy'})
            )


if __name__ == '__main__':
    model = XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss', random_state=123)
    min_val, max_val = 0, 5
    step = 0.5
    param_grid = {
        'alpha': np.arange(min_val, max_val + step, step),
        'lambda': np.arange(min_val, max_val + step, step),
        'gamma': np.arange(min_val, max_val + step, step)
    }
    name = 'gina_agnostic'  # 'prostate'
    X, y = prepare_datasets_for_classification({'gina_agnostic.csv': ('target', None, 100)})
    model = Regularization(model, param_grid)
    model.fit(X, y)
    search_results = model.get_search_results()
    search_results.to_excel(f'regularization_results_05_step_{name}.xlsx', index=False)