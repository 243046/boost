from time import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from category_encoders.cat_boost import CatBoostEncoder
from tune_sklearn import TuneSearchCV

from utils.metrics import metric_f1_score


class Classifier:
    def __init__(
            self,
            model,
            param_grid,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy', 'f1_score': make_scorer(metric_f1_score)}
    ):

        self.model = model
        self.param_grid = param_grid
        self.tuner = tuner
        self.tuner_scoring = tuner_scoring
        self.final_scoring = final_scoring

    def _make_pipeline(self):
        self.pipeline = Pipeline([
            ('encoder', CatBoostEncoder()),
            ('clf', self.model)
        ])

    def _add_prefix_to_grid(self, grid, prefix):
        new_keys = [f'{prefix}__{key}' for key in grid.keys()]
        return dict(zip(new_keys, grid.values()))

    def _make_cv(self, n_inner=5, n_outer=10):
        self.inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=123)
        self.outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=123)

    def _fit_clf(self, X, y):
        self._make_cv()
        grid = self._add_prefix_to_grid(self.param_grid, 'clf')
        if self.param_grid:
            clf = TuneSearchCV(self.pipeline,
                               param_distributions=grid,
                               search_optimization=self.tuner,
                               n_trials=15,
                               scoring=self.tuner_scoring,
                               cv=self.inner_cv,
                               early_stopping=False,
                               n_jobs=-1,
                               random_state=123
                               )
        else:
            clf = self.pipeline
        clf.fit(X, y)
        self.clf = clf

    def fit(self, X, y):
        self._make_pipeline()
        t0 = time()
        self._fit_clf(X, y)
        self.tuning_time_ = time()-t0 if self.param_grid else 0
        return self

    def cv_score(self, X, y):
        t0 = time()
        if self.param_grid:
            cv_results = cross_validate(self.clf.best_estimator_, X, y, scoring=self.final_scoring,
                                        cv=self.outer_cv, n_jobs=-1)
        else:
            cv_results = cross_validate(self.clf, X, y, scoring=self.final_scoring, cv=self.outer_cv, n_jobs=-1)
        evaluation = {k.lstrip('test_'): v for k, v in cv_results.items() if k.lstrip('test_') in self.final_scoring}
        final_eval_time, tuning_time = time()-t0, self.tuning_time_
        return evaluation, final_eval_time, tuning_time


class ClassifierRandomSearch(Classifier):
    def __init__(
            self,
            model,
            param_grid,
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy', 'f1_score': make_scorer(metric_f1_score)}
    ):

        super().__init__(model,
                         param_grid,
                         tuner=None,
                         tuner_scoring=tuner_scoring,
                         final_scoring=final_scoring
                         )

    def _fit_clf(self, X, y):
        self._make_cv()
        grid = self._add_prefix_to_grid(self.param_grid, 'clf')
        if self.param_grid:
            clf = RandomizedSearchCV(self.pipeline,
                                     param_distributions=grid,
                                     scoring=self.tuner_scoring,
                                     cv=self.inner_cv,
                                     n_jobs=-1,
                                     random_state=123
                                     )
        else:
            clf = self.pipeline
        clf.fit(X, y)
        self.clf = clf
