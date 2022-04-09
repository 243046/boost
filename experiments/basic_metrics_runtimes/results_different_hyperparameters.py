import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from wrappers.datasets_models_wrappers import DataModelsWrapper, DataModelsWrapperRandomSearch
from wrappers.datasets_models_wrappers_nlp import DataModelsWrapperNLP, DataModelsWrapperNLPRandomSearch
from data_processing.process_dataset import prepare_datasets_for_classification
from data_processing.process_dataset_nlp import prepare_nlp_for_classification

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


def run_nlp(param_dict, mode='randomized', tuner='hyperopt', scoring='accuracy',
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}):
    if mode == 'randomized':
        model = DataModelsWrapperNLPRandomSearch(param_dict, scoring=scoring, tfidf_kws=tfidf_kws)
    elif mode == 'TPE':
        model = DataModelsWrapperNLP(param_dict, tuner=tuner, scoring=scoring, tfidf_kws=tfidf_kws)
    model.fit()
    all_results = model.all_datasets_results_
    all_runtimes = model.all_datasets_runtimes_
    results_for_plotting = model.results_for_plotting_
    runtimes_for_plotting = model.runtimes_for_plotting_
    return all_results, all_runtimes, results_for_plotting, runtimes_for_plotting


if __name__ == '__main__':
    d = {
        'mushrooms.csv': ('class', 'all', None),
        'adult.csv': ('profit', [], None),
        'churn.csv': ('Churn', [], None),
        'creditcard.csv': ('Class', [], None),
        'prostate.csv': ('target', [], None)
    }

    X_1, y_1, X_2, y_2, X_3, y_3, X_4, y_4, X_5, y_5 = prepare_datasets_for_classification(d)

    params = {
        'n_estimators': [100],
        'max_depth': [5]
    }

    models = {
        'XGBoost': (XGBClassifier(use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), params),
        'XGBoost dart': (XGBClassifier(booster='dart', use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), params),
        'XGBoost approx': (XGBClassifier(tree_method='approx', use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), params),
        'XGBoost hist': (XGBClassifier(tree_method='hist', use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), params),
        'LightGBM': (LGBMClassifier(), params),
        'LightGBM no EFB': (LGBMClassifier(enable_bundle=False), params),
        'LightGBM GOSS': (LGBMClassifier(boosting_type='goss'), params),
        'LightGBM DART': (LGBMClassifier(boosting_type='dart'), params),
        'CatBoost': (CatBoostClassifier(verbose=False, random_state=123), params),
        'CatBoost langevin': (CatBoostClassifier(langevin=True, verbose=False, random_state=123), params)
    }

    param_dict = {
        'mushrooms': (X_1, y_1, models),
        'adult': (X_2, y_2, models),
        'churn': (X_3, y_3, models),
        'credit card': (X_4, y_4, models),
        'prostate': (X_5, y_5, models)
    }

    _, _, results_for_plotting, runtimes_for_plotting = run(param_dict=param_dict, mode='randomized', scoring='accuracy')

    name = 'hyperparameters'
    results_for_plotting.to_excel(f'../../results/results_{name}.xlsx', index=False)
    runtimes_for_plotting.to_excel(f'../../results/runtimes_{name}.xlsx', index=False)
