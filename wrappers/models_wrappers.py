from time import time

import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score

from models.classifiers import Classifier, ClassifierRandomSearch
from utils.metrics import metric_f1_score


class ModelsWrapper:
    def __init__(
            self,
            models,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')}
    ):

        self.models = models.copy()
        self.tuner = tuner
        self.tuner_scoring = tuner_scoring
        self.final_scoring = final_scoring

    def _score_single_model(self, X, y, model, param_grid):
        clf = Classifier(model, param_grid, tuner=self.tuner, tuner_scoring=self.tuner_scoring,
                         final_scoring=self.final_scoring)
        return clf.fit(X, y).cv_score(X, y)

    def _fit(self, X, y):
        results = {scoring: pd.DataFrame() for scoring in self.final_scoring}
        tuning_times, runtimes = pd.DataFrame(), pd.DataFrame()
        for model_name, (model, param_grid) in self.models.items():
            print(f'{model_name} in progress...')
            t0 = time()
            evaluation, final_eval_time, tuning_time = self._score_single_model(X, y, model, param_grid)
            for scoring in results:
                results[scoring][model_name] = evaluation[scoring]
            tuning_times[model_name] = [tuning_time]
            runtimes[model_name] = [final_eval_time]
            print(f'{model_name} done, took: {time()-t0:.3f}s')
        self.results_ = results
        self.tuning_times_ = tuning_times
        self.runtimes_ = runtimes

    def fit(self, X, y):
        self._fit(X, y)
        return self


class ModelsWrapperRandomSearch(ModelsWrapper):
    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierRandomSearch(model, param_grid, tuner_scoring=self.tuner_scoring,
                                     final_scoring=self.final_scoring)
        return clf.fit(X, y).cv_score(X, y)