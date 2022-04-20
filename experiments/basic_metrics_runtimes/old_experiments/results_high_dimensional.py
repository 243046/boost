import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy import stats

from data_processing.process_dataset import prepare_datasets_for_classification
from data_processing.process_dataset_nlp import prepare_nlp_for_classification
from experiments_runners.basic_runners import run, run_nlp
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    d = {
        'prostate.csv': ('target', [], None),
        'leukemia.csv': ('target', [], None),
        'weather_dataset.csv': ('target', [], 500)
    }

    X_1, y_1, X_2, y_2, X_3, y_3 = prepare_datasets_for_classification(d)

    X_4, y_4 = prepare_nlp_for_classification(
        dataset_name='imdb_dataset.csv',
        text_column='review_cleared',
        y_col='sentiment',
        nrows=3000
    )

    boosting_params = {
        'subsample': stats.uniform(0.5, 1.0)
    }
    xgb_params = {
        'reg_alpha': stats.loguniform(1, 10),
        'reg_lambda': stats.loguniform(1, 10)
    }
    lgbm_params = {
        'reg_alpha': stats.loguniform(1, 10),
        'reg_lambda': stats.loguniform(1, 10)
    }
    catboost_params = {
        'reg_lambda': stats.loguniform(1, 10)
    }

    models = {
        'Gradient Boosting': (GradientBoostingClassifier(), boosting_params),
        'XGBoost': (XGBClassifier(use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), xgb_params),
        'LightGBM': (LGBMClassifier(), lgbm_params),
        'CatBoost': (CatBoostClassifier(n_estimators=100, verbose=False, random_state=123), catboost_params)
    }

    param_dict = {
        'prostate': (X_1, y_1, models),
        'leukemia': (X_2, y_2, models),
        'weather': (X_3, y_3, models)
    }

    param_dict_nlp = {
        'IMDB reviews': (X_4, y_4, models)
    }

    tfidf_kws = {'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}

    all_results, all_runtimes, results_for_plotting, runtimes_for_plotting = run(param_dict=param_dict,
                                                                                 mode='randomized')

    _, _, results_for_plotting_nlp, runtimes_for_plotting_nlp = run_nlp(param_dict=param_dict_nlp,
                                                                        mode='randomized',
                                                                        tfidf_kws=tfidf_kws
                                                                        )

    name = 'high_dimensional'
    all_results = pd.concat([results_for_plotting, results_for_plotting_nlp])
    all_runtimes = pd.concat([runtimes_for_plotting, runtimes_for_plotting_nlp])
    all_results.to_excel(f'../../results/results_{name}.xlsx', index=False)
    all_runtimes.to_excel(f'../../results/runtimes_{name}.xlsx', index=False)