from time import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from category_encoders.cat_boost import CatBoostEncoder
from catboost import CatBoostClassifier

from models.classifiers import Classifier, ClassifierRandomSearch
from wrappers.models_wrappers import ModelsWrapper, ModelsWrapperRandomSearch
from wrappers.datasets_models_wrappers import DataModelsWrapper, DataModelsWrapperRandomSearch
from utils.metrics import metric_f1_score


class ClassifierCategorical(Classifier):
    def _fit_clf(self, X, y):
        self._make_cv()
        clf = self.model
        clf.fit(X, y)
        self.clf = clf


class ClassifierCategoricalRandomSearch(ClassifierRandomSearch):
    def _make_pipeline(self):
        self.pipeline = self.clf


class ModelsWrapperCategorical(ModelsWrapper):
    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierCategorical(model, param_grid, tuner=self.tuner, tuner_scoring=self.tuner_scoring)
        return clf.fit(X, y).cv_score(X, y)


class ModelsWrapperCategoricalRandomSearch(ModelsWrapperRandomSearch):
    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierCategoricalRandomSearch(model, param_grid, tuner_scoring=self.tuner_scoring)
        return clf.fit(X, y).cv_score(X, y)


class DataModelsWrapperCategorical(DataModelsWrapper):
    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperCategorical(models, tuner=self.tuner, tuner_scoring=self.tuner_scoring)
        model.fit(X, y)
        return model.results_, model.tuning_times_, model.runtimes_


class DataModelsWrapperCategoricalRandomSearch(DataModelsWrapperRandomSearch):
    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperCategoricalRandomSearch(models, tuner_scoring=self.tuner_scoring)
        model.fit(X, y)
        return model.results_, model.tuning_times_, model.runtimes_


class CatBoostExperiment:
    def __init__(self, models=None, final_scoring={'accuracy': 'accuracy', 'f1_score': make_scorer(metric_f1_score)},
                 dataset_name='mushrooms'):
        self.models = models
        self.final_scoring = final_scoring
        self.dataset_name = dataset_name

    def _make_cv(self, n_inner=5, n_outer=10):
        self.inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=123)
        self.outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=123)

    def _fit(self, X, y):
        results = {scoring: pd.DataFrame() for scoring in self.final_scoring}
        runtimes = pd.DataFrame()
        self._make_cv()
        if self.models is None:
            catboost_catboost_encoder = Pipeline([
                ('encoder', CatBoostEncoder()),
                ('CatBoost', CatBoostClassifier(n_estimators=100, verbose=False, random_state=321))
            ])
            catboost_embedded = CatBoostClassifier(n_estimators=100, verbose=False, random_state=123)
            catboost_ohe = CatBoostClassifier(n_estimators=100, verbose=False, random_state=123, one_hot_max_size=99999)
            self.models = {
                'CatBoost preprocessed': catboost_catboost_encoder,
                'CatBoost embedded': catboost_embedded,
                'CatBoost OHE': catboost_ohe
            }
        for model_name, model in self.models.items():
            print(model_name)
            cat_features = [X.columns.get_loc(col) for col in X]
            t0 = time()
            if model_name == 'CatBoost embedded':
                cv_results = cross_validate(model, X, y, scoring=self.final_scoring, cv=self.outer_cv, n_jobs=-1,
                                            fit_params={'cat_features': cat_features})
            else:
                cv_results = cross_validate(model, X, y, scoring=self.final_scoring, cv=self.outer_cv, n_jobs=-1)
            t1 = time()
            evaluation = {k.lstrip('test_'): v for k, v in cv_results.items() if k.lstrip('test_') in self.final_scoring}
            for scoring in results:
                results[scoring][model_name] = evaluation[scoring]
            runtimes[model_name] = [t1-t0]
        for scoring in results:
            results[scoring]['dataset'] = self.dataset_name
        runtimes['dataset'] = self.dataset_name
        self.results_, self.runtimes_ = results, runtimes

    def fit(self, X, y):
        self._fit(X, y)
        return self