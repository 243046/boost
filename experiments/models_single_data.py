import warnings

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from scipy import stats

from wrappers.models_wrappers import ModelsWrapper, ModelsWrapperRandomSearch
warnings.filterwarnings('ignore')


def run(X, y, models, mode='randomized', tuner='hyperopt', scoring='accuracy'):
    if mode == 'randomized':
        model = ModelsWrapperRandomSearch(models, scoring=scoring)
    elif mode == 'TPE':
        model = ModelsWrapper(models, tuner=tuner, scoring=scoring)
    model.fit(X, y)
    results = model.results_
    runtimes = model.runtimes_
    return results, runtimes


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5)

    boosting_params = {
    'n_estimators': [50, 100],
    }
    xgb_params = {
        'n_estimators': [50, 100],
       # 'max_depth': [4, 5],
       # 'learning_rate': [0.05, 0.1]
    }
    lgbm_params = {
        'n_estimators': [50, 100],
       # 'max_depth': [4, 5],
       # 'learning_rate': [0.05, 0.1]
    }
    catboost_params = {
        'iterations': [50, 100],
       # 'depth': [4, 5],
       # 'learning_rate': [0.05, 0.1]
    }

    models = {
        'Gradient Boosting': (GradientBoostingClassifier(), boosting_params),
        'XGBoost': (XGBClassifier(use_label_encoder=False,
                                 eval_metric='logloss', random_state=123), xgb_params),
        'LightGBM': (LGBMClassifier(), lgbm_params),
        'CatBoost': (CatBoostClassifier(verbose=False, random_state=123), catboost_params)
    }

    results, runtimes = run(X, y, models, mode='randomized', scoring='accuracy')
