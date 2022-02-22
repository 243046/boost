import warnings
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from scipy import stats

from models.classifiers import Classifier, ClassifierRandomSearch
warnings.filterwarnings('ignore')


def run(X, y, model, param_grid, mode='randomized', tuner='hyperopt', scoring='accuracy'):
    if mode == 'randomized':
        model = ClassifierRandomSearch(model, param_grid, scoring=scoring)
    elif mode == 'TPE':
        model = Classifier(model, param_grid, tuner=tuner, scoring=scoring)
    model.fit(X, y)
    return model.score(X, y)


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5)
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5],
        'learning_rate': stats.loguniform(0.01, 0.1)
    }

    score = run(X, y, model, param_grid, mode='randomized', scoring='accuracy')
    print(f'Cross-val score: {score:.3f}')
