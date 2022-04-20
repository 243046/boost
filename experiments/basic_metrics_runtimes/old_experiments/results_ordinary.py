import warnings

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy import stats

from data_processing.process_dataset import prepare_datasets_for_classification
from experiments_runners.basic_runners import run
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    d = {
        'mushrooms.csv': ('class', 'all', None),
        'adult.csv': ('profit', [], None),
        'churn.csv': ('Churn', [], None),
        'creditcard.csv': ('Class', [], None)
    }

    X_1, y_1, X_2, y_2, X_3, y_3, X_4, y_4 = prepare_datasets_for_classification(d)

    boosting_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': stats.loguniform(0.01, 0.1)
    }
    xgb_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': stats.loguniform(0.01, 0.1)
    }
    lgbm_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': stats.loguniform(0.01, 0.1)
    }
    catboost_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': stats.loguniform(0.01, 0.1)
    }

    models = {
        'Gradient Boosting': (GradientBoostingClassifier(), boosting_params),
        'XGBoost': (XGBClassifier(use_label_encoder=False,
                                 eval_metric='logloss', random_state=123), xgb_params),
        'LightGBM': (LGBMClassifier(), lgbm_params),
        'CatBoost': (CatBoostClassifier(verbose=False, random_state=123), catboost_params)
    }

    param_dict = {
        'mushrooms': (X_1, y_1, models),
        'adult': (X_2, y_2, models),
        'churn': (X_3, y_3, models),
        'credit card': (X_4, y_4, models)
    }

    all_results, all_runtimes, results_for_plotting, runtimes_for_plotting = run(param_dict=param_dict,
                                                                                 mode='randomized')

    name = 'ordinary'
    results_for_plotting.to_excel(f'../../results/results_{name}.xlsx', index=False)
    runtimes_for_plotting.to_excel(f'../../results/runtimes_{name}.xlsx', index=False)
