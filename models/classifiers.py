from time import time

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from tune_sklearn import TuneSearchCV

from data_processing.cat_encoder import PermutedCatBoostEncoder
from utils.metrics import metric_f1_score


class Classifier:
    """
    Base class in the basic experiment with TPE tuning.
    It takes a base model, fits it to the given data and produces results.

    Parameters
    ----------
    model : scikit-learn-type classifier
        A base model, e.g. GradientBoostingClassifier().

    param_grid : dict
        A search space which serves as an input to TPE search. If an empty dict
        is given, then no hyperparameter tuning is performed.

    tuner : string, default='hyperopt'
        Dependency needed for the tuning - hyperopt package is used by default.

    tuner_scoring : string, default='neg_log_loss'
        A metric which is used in the hyperparameter search.

    final_scoring : dict
        A dict of metrics which will be used in model evaluation.

    Attributes
    ----------
    tuning_time_ : float
        The time elapsed during hyperparameter tuning. If no tuning was performed, then
        the value of the attribute is 0.

    See Also
    --------
    ClassifierRandomSearch : Classifier with randomized search instead of TPE search.
    ClassifierNLP : Classifier which processes text (NLP) data.
    """

    def __init__(
            self,
            model,
            param_grid,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')}
    ):

        self.model = model
        self.param_grid = param_grid
        self.tuner = tuner
        self.tuner_scoring = tuner_scoring
        self.final_scoring = final_scoring

    def _make_pipeline(self):
        self.pipeline = Pipeline([
            ('encoder', PermutedCatBoostEncoder()),
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
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like
            Feature matrix to be fitted.

        y : array-like, one-dimensional
            Class labels to be used during training.
        """

        self._make_pipeline()
        t0 = time()
        self._fit_clf(X, y)
        self.tuning_time_ = time()-t0 if self.param_grid else 0
        return self

    def cv_score(self, X, y):
        """
        Calculate cross-validation scores according to final_scoring dictionary.

        Parameters
        ----------
        X : array-like
            Feature matrix to be fitted.

        y : array-like, one-dimensional
            Class labels to be used during training.

        Returns
        ----------
        evaluation : dict
            Dict with cross-validation metrics names and scores.

        final_eval_time : float
            Runtime of the algorithm (e.g. time to evaluate it using 10-fold cv).

        tuning_time : float
            The time elapsed during hyperparameter tuning. If no tuning was performed, then
            the value of the attribute is 0.
        ----------
        """

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
    """The randomized search version of the Classifier class"""
    def __init__(
            self,
            model,
            param_grid,
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')}
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
                                     n_iter=30,
                                     scoring=self.tuner_scoring,
                                     cv=self.inner_cv,
                                     n_jobs=-1,
                                     random_state=123
                                     )
        else:
            clf = self.pipeline
        clf.fit(X, y)
        self.clf = clf
