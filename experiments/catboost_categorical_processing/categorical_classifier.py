from time import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders.cat_boost import CatBoostEncoder
from catboost import CatBoostClassifier

from data_processing.cat_encoder import PermutedCatBoostEncoder
from utils.metrics import metric_f1_score


class CatBoostExperiment:
    def __init__(
            self,
            models=None,
            final_scoring={'accuracy': 'accuracy',
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted'),
                           'log loss': 'neg_log_loss'},
            dataset_name='mushrooms'
    ):

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
            different_permutations = Pipeline([
                ('encoder', PermutedCatBoostEncoder()),
                ('CatBoost', CatBoostClassifier(boosting_type='Ordered', n_estimators=100,
                                                verbose=False, random_state=321))
            ])
            identical_permutations_ord = CatBoostClassifier(boosting_type='Ordered', n_estimators=100,
                                                            verbose=False, random_state=123)
            embedded_plain = CatBoostClassifier(boosting_type='Plain', n_estimators=100,
                                                              verbose=False, random_state=123)
            ohe = Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ('CatBoost', CatBoostClassifier(boosting_type='Ordered', n_estimators=100,
                                                verbose=False, random_state=321))
            ])
            self.models = {
                'different\nperm.': different_permutations,
                'embedded\nidentical perm.': identical_permutations_ord,
                'embedded\nPlain': embedded_plain,
                'OHE': ohe
            }
        for model_name, model in self.models.items():
            print(model_name)
            cat_features = [X.columns.get_loc(col) for col in X]
            t0 = time()
            if model_name in ['embedded\nidentical perm.', 'embedded\nPlain']:
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