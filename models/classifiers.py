from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from category_encoders.cat_boost import CatBoostEncoder
from tune_sklearn import TuneSearchCV
import pandas as pd

# from data_processing.cat_encoder import PermutedCatBoostEncoder
from utils.timer import timeit


class Classifier:
    def __init__(
            self,
            model,
            param_grid,
            tuner='hyperopt',
            scoring='accuracy'
    ):

        self.model = model
        self.param_grid = param_grid
        self.tuner = tuner
        self.scoring = scoring

    def _make_pipeline(self):
        self.pipeline = Pipeline([
           # ('imputer', KNNImputer()),
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
                               scoring=self.scoring,
                               cv=self.inner_cv,
                               early_stopping=False,
                               n_jobs=-1,
                               random_state=123
                               )
        else:
            clf = self.pipeline
        clf.fit(X, y)
        print(clf.best_estimator_)
        self.clf = clf

    @timeit
    def fit(self, X, y):
        self._make_pipeline()
        self._fit_clf(X, y)
        return self

    def cv_score(self, X, y):
        if self.param_grid:
            return cross_val_score(self.clf.best_estimator_, X, y, scoring=self.scoring, cv=self.outer_cv, n_jobs=-1)
        else:
            return cross_val_score(self.clf, X, y, scoring=self.scoring, cv=self.outer_cv, n_jobs=-1)

    def score(self, X, y):
        return self.cv_score(X, y).mean()


class ClassifierRandomSearch(Classifier):
    def __init__(self, model, param_grid, scoring='accuracy'):
        super().__init__(model,
                         param_grid,
                         tuner=None,
                         scoring=scoring
                         )

    def _fit_clf(self, X, y):
        self._make_cv()
        grid = self._add_prefix_to_grid(self.param_grid, 'clf')
        if self.param_grid:
            clf = RandomizedSearchCV(self.pipeline,
                                     param_distributions=grid,
                                     scoring=self.scoring,
                                     cv=self.inner_cv,
                                     n_jobs=-1,
                                     random_state=123
                                     )
        else:
            clf = self.pipeline
        clf.fit(X, y)
        print(clf.best_estimator_)
        pd.DataFrame(clf.cv_results_).to_excel('res.xlsx', index=False)
        self.clf = clf