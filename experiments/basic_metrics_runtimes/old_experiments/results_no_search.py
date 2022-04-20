import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from data_processing.process_dataset import prepare_datasets_for_classification
from data_processing.process_dataset_nlp import prepare_nlp_for_classification
from experiments_runners.basic_runners import run, run_nlp

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    d = {
        'mushrooms.csv': ('class', 'all', 100)
    }

    X_1, y_1 = prepare_datasets_for_classification(d)

    X_8, y_8 = prepare_nlp_for_classification(
        dataset_name='imdb_dataset.csv',
        text_column='review_cleared',
        y_col='sentiment',
        nrows=200
    )

    models = {
        'Gradient Boosting': (GradientBoostingClassifier(), {}),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                  random_state=123), {}),
        'LightGBM': (LGBMClassifier(), {}),
        'CatBoost': (CatBoostClassifier(n_estimators=100, verbose=False,
                                        random_state=123), {})
    }

    param_dict = {
        'mushrooms': (X_1, y_1, models)
    }

    param_dict_nlp = {
        'IMDB reviews': (X_8, y_8, models)
    }

    tfidf_kws = {'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}

    # d = {
    #     'mushrooms.csv': ('class', 'all', None),
    #     'adult.csv': ('profit', [], None),
    #     'churn.csv': ('Churn', [], None),
    #     'creditcard.csv': ('Class', [], None),
    #     'prostate.csv': ('target', [], None),
    #     'leukemia.csv': ('target', [], None),
    #     'weather_dataset.csv': ('target', [], 200)
    # }
    #
    # X_1, y_1, X_2, y_2, X_3, y_3, X_4, y_4, X_5, y_5, X_6, y_6, X_7, y_7 = prepare_datasets_for_classification(d)
    #
    # X_8, y_8 = prepare_nlp_for_classification(
    #     dataset_name='imdb_dataset.csv',
    #     text_column='review_cleared',
    #     y_col='sentiment',
    #     nrows=2000
    # )
    #
    # models = {
    #     'Gradient Boosting': (GradientBoostingClassifier(), {}),
    #     'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss',
    #                               random_state=123), {}),
    #     'LightGBM': (LGBMClassifier(), {}),
    #     'CatBoost': (CatBoostClassifier(n_estimators=100, verbose=False,
    #                                     random_state=123), {})
    # }
    #
    # param_dict = {
    #     'mushrooms': (X_1, y_1, models),
    #     'adult': (X_2, y_2, models),
    #     'churn': (X_3, y_3, models),
    #     'credit card': (X_4, y_4, models),
    #     'prostate': (X_5, y_5, models),
    #     'leukemia': (X_6, y_6, models),
    #     'weather': (X_7, y_7, models)
    # }

    param_dict_nlp = {
        'IMDB reviews': (X_8, y_8, models)
    }

    tfidf_kws = {'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}

    all_results, all_tuning_times, all_runtimes = run(param_dict=param_dict, mode='randomized')

    all_results_nlp, all_tuning_times_nlp, all_runtimes_nlp = run_nlp(param_dict=param_dict_nlp,
                                                                        mode='randomized',
                                                                        tfidf_kws=tfidf_kws
                                                                        )

    for scoring in all_results:
        all_results[scoring] = pd.concat([all_results[scoring], all_results_nlp[scoring]])
    all_tuning_times = pd.concat([all_tuning_times, all_tuning_times_nlp])
    all_runtimes = pd.concat([all_runtimes, all_runtimes_nlp])
    # all_results.to_excel('../../results/results_no_search.xlsx', index=False)
    # all_runtimes.to_excel('../../results/runtimes_no_search.xlsx', index=False)
