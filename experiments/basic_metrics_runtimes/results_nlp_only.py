import warnings

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy import stats

from wrappers.datasets_models_wrappers_nlp import DataModelsWrapperNLP, DataModelsWrapperNLPRandomSearch
from data_processing.process_dataset_nlp import prepare_nlp_for_classification
warnings.filterwarnings('ignore')


def run(param_dict, mode='randomized', tuner='hyperopt', scoring='accuracy',
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
    X, y = prepare_nlp_for_classification(
        dataset_name='imdb_dataset.csv',
        text_column='review_cleared',
        y_col='sentiment',
        nrows=1000
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
        'IMDB reviews': (X, y, models)
    }

    tfidf_kws = {'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}

    all_results, all_runtimes, results_for_plotting, runtimes_for_plotting = run(param_dict=param_dict,
                                                                                 mode='randomized',
                                                                                 scoring='accuracy',
                                                                                 tfidf_kws=tfidf_kws
                                                                                 )

    results_for_plotting.to_excel('../../results/nlp_results.xlsx', index=False)
    runtimes_for_plotting.to_excel('../../results/nlp_runtimes.xlsx', index=False)
