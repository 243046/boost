import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from scipy import stats

from wrappers.datasets_models_wrappers import DataModelsWrapper, DataModelsWrapperRandomSearch
from data_processing.process_dataset import prepare_datasets_for_classification
warnings.filterwarnings('ignore')


def run(param_dict, mode='randomized', tuner='hyperopt', scoring='accuracy'):
    if mode == 'randomized':
        model = DataModelsWrapperRandomSearch(param_dict, scoring=scoring)
    elif mode == 'TPE':
        model = DataModelsWrapper(param_dict, tuner=tuner, scoring=scoring)
    model.fit()
    all_results = model.all_datasets_results_
    all_runtimes = model.all_datasets_runtimes_
    results_for_plotting = model.results_for_plotting_
    runtimes_for_plotting = model.runtimes_for_plotting_
    return all_results, all_runtimes, results_for_plotting, runtimes_for_plotting


if __name__ == '__main__':
    #X_1, y_1 = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5)
    #X_1, y_1 = X_1[:100, :], y_1[:100]
    d = {
        'mushrooms.csv': ('class', [], 100),
        'mushrooms1.csv': ('class', [], 200)
    }

    X_1, y_1, X_2, y_2 = prepare_datasets_for_classification(d)

    boosting_params_1 = {
    'n_estimators': [50, 100],
    }
    xgb_params_1 = {
        'n_estimators': [50, 100],
       # 'max_depth': [4, 5],
       # 'learning_rate': [0.05, 0.1]
    }
    lgbm_params_1 = {
        'n_estimators': [50, 100],
       # 'max_depth': [4, 5],
       # 'learning_rate': [0.05, 0.1]
    }
    catboost_params_1 = {
        'iterations': [50, 100],
       # 'depth': [4, 5],
       # 'learning_rate': [0.05, 0.1]
    }

    models_1 = {
        'Gradient Boosting': (GradientBoostingClassifier(), boosting_params_1),
        'XGBoost': (XGBClassifier(use_label_encoder=False,
                                 eval_metric='logloss', random_state=123), xgb_params_1),
        'LightGBM': (LGBMClassifier(), lgbm_params_1),
        'CatBoost': (CatBoostClassifier(verbose=False, random_state=123), catboost_params_1)
    }

    #X_2, y_2 = make_classification()

    boosting_params_2 = {
        'n_estimators': [50, 100],
    }
    xgb_params_2 = {
        'n_estimators': [50, 100],
        # 'max_depth': [4, 5],
        # 'learning_rate': [0.05, 0.1]
    }
    lgbm_params_2 = {
        'n_estimators': [50, 100],
        # 'max_depth': [4, 5],
        # 'learning_rate': [0.05, 0.1]
    }
    catboost_params_2 = {
        'iterations': [50, 100],
        # 'depth': [4, 5],
        # 'learning_rate': [0.05, 0.1]
    }

    models_2 = {
        'Gradient Boosting': (GradientBoostingClassifier(), boosting_params_2),
        'XGBoost': (XGBClassifier(use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), xgb_params_2),
        'LightGBM': (LGBMClassifier(), lgbm_params_2),
        'CatBoost': (CatBoostClassifier(verbose=False, random_state=123), catboost_params_2)
    }

    param_dict = {
        'dataset_1': (X_1, y_1, models_1),
        'dataset_2': (X_2, y_2, models_2)
    }

    all_results, all_runtimes, results_for_plotting, runtimes_for_plotting = run(param_dict=param_dict,
                                                                                 mode='randomized', scoring='accuracy')

    # results_for_plotting.to_excel('../results/all_results.xlsx', index=False)
    # runtimes_for_plotting.to_excel('../results/all_runtimes.xlsx', index=False)
