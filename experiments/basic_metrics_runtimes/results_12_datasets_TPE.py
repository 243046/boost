import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ray import tune

from data_processing.process_dataset import prepare_datasets_for_classification
from data_processing.process_dataset_nlp import prepare_nlp_for_classification
from experiments_runners.basic_runners import run, run_nlp


if __name__ == '__main__':
    d = {
        'adult_full.csv': ('profit', [], None),
        'heart.csv': ('target', ['cp', 'restecg'], None),
        'amazon.csv': ('ACTION', 'all', None),
        'mushrooms.csv': ('class', 'all', None),
        'breast_cancer.csv': ('target', [], None),
        'churn.csv': ('Churn', [], None),
        'creditcard_full.csv': ('Class', [], 30000),
        'prostate.csv': ('target', [], None),
        'leukemia.csv': ('target', [], None),
        'gina_agnostic.csv': ('target', [], None),
        'weather_dataset_full.csv': ('target', [], 500),
    }

    X_1, y_1, X_2, y_2, X_3, y_3, X_4, y_4, X_5, y_5, X_6, y_6, X_7, y_7, X_8, y_8,\
        X_9, y_9, X_10, y_10, X_11, y_11 = prepare_datasets_for_classification(d)

    X_12, y_12 = prepare_nlp_for_classification(
        dataset_name='imdb_dataset_full.csv',
        text_column='review_cleared',
        y_col='sentiment',
        nrows=2000
    )

    boosting_params_default = {
        'max_depth': [2, 3, 4, 5, 8, 10],
        'learning_rate': tune.loguniform(0.01, 0.3),
        'min_samples_split': [2, 5, 10]
    }
    xgb_params_default = {
        'max_depth': [2, 3, 4, 5, 8, 10],
        'learning_rate': tune.loguniform(0.01, 0.3),
        'gamma': tune.uniform(0, 3),
        'alpha': tune.uniform(0, 1),
        'lambda': tune.uniform(0, 3)
    }
    lgbm_params_default = {
        'learning_rate': tune.loguniform(0.01, 0.3),
        'num_leaves': [3, 7, 1, 31, 127],
        'top_rate': tune.uniform(0.1, 0.5),
        'other_rate': tune.uniform(0.05, 0.2),
        'reg_alpha': tune.uniform(0, 1),
        'reg_lambda': tune.uniform(0, 3)
    }
    catboost_params_default = {
        'max_depth': [2, 3, 4, 5, 8, 10],
        'l2_leaf_reg': tune.uniform(0, 5),
        'leaf_estimation_iterations': [1, 10]
    }

    n_estimators_default = 150
    subsample_default = 0.75
    colsample_by_node_default = 0.6

    boosting_init_default = {
        'n_estimators': n_estimators_default,
        'subsample': subsample_default,
        'max_features': colsample_by_node_default
    }
    xgb_init_default = {
        'n_estimators': n_estimators_default,
        'subsample': subsample_default,
        'colsample_by_node': colsample_by_node_default
    }
    lgbm_init_default = {
        'boosting_type': 'goss',
        'n_estimators': n_estimators_default,
        'subsample': subsample_default,
        'colsample_by_node': colsample_by_node_default
    }
    catboost_init_default = {
        'boosting_type': 'Ordered',
        'n_estimators': n_estimators_default,
        'subsample': subsample_default,
        'colsample_bylevel': colsample_by_node_default
    }

    models_default = {
        'Gradient Boosting': (GradientBoostingClassifier(**boosting_init_default), boosting_params_default),
        'XGBoost': (XGBClassifier(**xgb_init_default, use_label_encoder=False,
                                 eval_metric='logloss', random_state=123), xgb_params_default),
        'LightGBM': (LGBMClassifier(**lgbm_init_default), lgbm_params_default),
        'CatBoost': (CatBoostClassifier(**catboost_init_default, verbose=False, random_state=123), catboost_params_default)
    }

    boosting_init_default_small_rows = boosting_init_default | {'subsample': 1}
    xgb_init_default_small_rows = xgb_init_default | {'subsample': 1}
    lgbm_init_default_small_rows = lgbm_init_default | {'subsample': 1}
    catboost_init_default_small_rows = catboost_init_default | {'subsample': 1}

    models_default_small_rows = {
        'Gradient Boosting': (GradientBoostingClassifier(**boosting_init_default_small_rows), boosting_params_default),
        'XGBoost': (XGBClassifier(**xgb_init_default_small_rows, use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), xgb_params_default),
        'LightGBM': (LGBMClassifier(**lgbm_init_default_small_rows), lgbm_params_default),
        'CatBoost': (CatBoostClassifier(**catboost_init_default_small_rows, verbose=False, random_state=123), catboost_params_default)
    }

    boosting_params_reg = {
        'max_depth': [2, 3, 4, 5, 8, 10],
        'learning_rate': tune.loguniform(0.01, 0.3),
        'min_samples_split': [2, 5, 10]
    }
    xgb_params_reg = {
        'max_depth': [2, 3, 4, 5, 8, 10],
        'learning_rate': tune.loguniform(0.01, 0.3),
        'gamma': tune.uniform(0, 10),
        'alpha': tune.uniform(0, 5),
        'lambda': tune.uniform(0, 10)
    }
    lgbm_params_reg = {
        'learning_rate': tune.loguniform(0.01, 0.3),
        'num_leaves': [3, 7, 1, 31, 127],
        'top_rate': tune.uniform(0.1, 0.5),
        'other_rate': tune.uniform(0.05, 0.2),
        'reg_alpha': tune.uniform(0, 5),
        'reg_lambda': tune.uniform(0, 10)
    }
    catboost_params_reg = {
        'max_depth': [2, 3, 4, 5, 8, 10],
        'l2_leaf_reg': tune.uniform(0, 12),
        'leaf_estimation_iterations': [1, 10]
    }

    n_estimators_reg_microarray = 150
    subsample_reg_microarray = 1
    colsample_by_node_reg_microarray = 0.4

    boosting_init_reg_microarray = {
        'n_estimators': n_estimators_reg_microarray,
        'subsample': subsample_reg_microarray,
        'max_features': colsample_by_node_reg_microarray
    }
    xgb_init_reg_microarray = {
        'n_estimators': n_estimators_reg_microarray,
        'subsample': subsample_reg_microarray,
        'colsample_by_node': colsample_by_node_reg_microarray
    }
    lgbm_init_reg_microarray = {
        'boosting_type': 'goss',
        'n_estimators': n_estimators_reg_microarray,
        'subsample': subsample_reg_microarray,
        'colsample_by_node': colsample_by_node_reg_microarray
    }
    catboost_init_reg_microarray = {
        'boosting_type': 'Ordered',
        'n_estimators': n_estimators_reg_microarray,
        'subsample': subsample_reg_microarray,
        'colsample_bylevel': colsample_by_node_reg_microarray
    }

    models_reg_microarray = {
        'Gradient Boosting': (GradientBoostingClassifier(**boosting_init_reg_microarray), boosting_params_reg),
        'XGBoost': (XGBClassifier(**xgb_init_reg_microarray, use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), xgb_params_reg),
        'LightGBM': (LGBMClassifier(**lgbm_init_reg_microarray), lgbm_params_reg),
        'CatBoost': (
        CatBoostClassifier(**catboost_init_reg_microarray, verbose=False, random_state=123), catboost_params_reg)
    }

    n_estimators_reg_image_nlp = 50
    subsample_reg_image_nlp = 0.5
    colsample_by_node_reg_image_nlp = 0.4

    boosting_init_reg_image_nlp = {
        'n_estimators': n_estimators_reg_image_nlp,
        'subsample': subsample_reg_image_nlp,
        'max_features': colsample_by_node_reg_image_nlp
    }
    xgb_init_reg_image_nlp = {
        'n_estimators': n_estimators_reg_image_nlp,
        'subsample': subsample_reg_image_nlp,
        'colsample_by_node': colsample_by_node_reg_image_nlp
    }
    lgbm_init_reg_image_nlp = {
        'boosting_type': 'goss',
        'n_estimators': n_estimators_reg_image_nlp,
        'subsample': subsample_reg_image_nlp,
        'colsample_by_node': colsample_by_node_reg_image_nlp
    }
    catboost_init_reg_image_nlp = {
        'boosting_type': 'Plain',
        'n_estimators': n_estimators_reg_image_nlp,
        'subsample': subsample_reg_image_nlp,
        'colsample_bylevel': colsample_by_node_reg_image_nlp
    }

    models_reg_image_nlp = {
        'Gradient Boosting': (GradientBoostingClassifier(**boosting_init_reg_image_nlp), boosting_params_reg),
        'XGBoost': (XGBClassifier(**xgb_init_reg_image_nlp, use_label_encoder=False,
                                  eval_metric='logloss', random_state=123), xgb_params_reg),
        'LightGBM': (LGBMClassifier(**lgbm_init_reg_image_nlp), lgbm_params_reg),
        'CatBoost': (
            CatBoostClassifier(**catboost_init_reg_image_nlp, verbose=False, random_state=123), catboost_params_reg)
    }

    param_dict = {
        'adult_full.csv': (X_1, y_1, models_default.copy()),
        'heart.csv': (X_2, y_2, models_default_small_rows.copy()),
        'amazon.csv': (X_3, y_3, models_default.copy()),
        'mushrooms.csv': (X_4, y_4, models_default.copy()),
        'breast_cancer.csv': (X_5, y_5, models_default_small_rows.copy()),
        'churn.csv': (X_6, y_6, models_default.copy()),
        'creditcard_full.csv': (X_7, y_7, models_default.copy()),
        'prostate.csv': (X_8, y_8, models_reg_microarray.copy()),
        'leukemia.csv': (X_9, y_9, models_reg_microarray.copy()),
        'gina_agnostic.csv': (X_10, y_10, models_reg_image_nlp.copy()),
        'weather_dataset_full.csv': (X_11, y_11, models_reg_image_nlp.copy())
    }

    param_dict_nlp = {
        'IMDB reviews': (X_12, y_12, models_reg_image_nlp.copy())
    }

    results, tuning_times, runtimes = run(param_dict=param_dict, mode='TPE')

    results_nlp, tuning_times_nlp, runtimes_nlp = run_nlp(param_dict=param_dict_nlp, mode='TPE')

    name = '12_datasets_TPE'
    for scoring in runtimes:
        path_to_save = f'../results_colab/results_{scoring}_{name}.xlsx'
        pd.concat([results[scoring], results_nlp[scoring]]).to_excel(path_to_save, index=False)
    all_tuning_times = pd.concat([tuning_times, tuning_times_nlp])
    all_runtimes = pd.concat([runtimes, runtimes_nlp])
    all_tuning_times.to_excel(f'../results_colab/tuning_times_{name}.xlsx', index=False)
    all_runtimes.to_excel(f'../results_colab/runtimes_{name}.xlsx', index=False)