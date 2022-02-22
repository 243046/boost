from time import time

import pandas as pd

from models.classifiers import Classifier, ClassifierRandomSearch


class ModelsWrapper:
    def __init__(self, models, tuner='hyperopt', scoring='accuracy'):
        self.models = models.copy()
        self.tuner = tuner
        self.scoring = scoring

    def _score_single_model(self, X, y, model, param_grid):
        clf = Classifier(model, param_grid, tuner=self.tuner, scoring=self.scoring)
        clf.fit(X, y)
        return clf.cv_score(X, y)

    def _fit(self, X, y):
        results = pd.DataFrame()
        runtimes = pd.DataFrame()
        for model_name, (model, param_grid) in self.models.items():
            t0 = time()
            results[model_name] = self._score_single_model(X, y, model, param_grid)
            runtimes[model_name] = [time() - t0]
        self.results_ = results
        self.runtimes_ = runtimes

    def fit(self, X, y):
        self._fit(X, y)
        return self


class ModelsWrapperRandomSearch(ModelsWrapper):
    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierRandomSearch(model, param_grid, scoring=self.scoring)
        clf.fit(X, y)
        return clf.cv_score(X, y)