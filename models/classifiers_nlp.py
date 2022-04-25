from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline

from models.classifiers import Classifier
from utils.metrics import metric_f1_score


class ClassifierNLP(Classifier):
    def __init__(
            self,
            model,
            param_grid,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')},
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 3000},
            svd_kws={'n_components': 100}):

        super().__init__(model=model,
                         param_grid=param_grid,
                         tuner=tuner,
                         tuner_scoring=tuner_scoring,
                         final_scoring=final_scoring
                         )

        self.tfidf_kws = tfidf_kws
        self.svd_kws = svd_kws

    def _make_pipeline(self):
        self.pipeline = Pipeline([
            ('tf-idf', TfidfVectorizer(**self.tfidf_kws)),
          #  ('svd', TruncatedSVD(**self.svd_kws)),
            ('clf', self.model)
        ])


class ClassifierNLPRandomSearch(ClassifierNLP):
    def __init__(
            self,
            model,
            param_grid,
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')},
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 3000},
            svd_kws={'n_components': 100}
    ):

        super().__init__(model=model, param_grid=param_grid, tuner=None, tuner_scoring=tuner_scoring,
                         final_scoring=final_scoring, tfidf_kws=tfidf_kws, svd_kws=svd_kws)

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