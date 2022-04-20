from sklearn.metrics import make_scorer

from wrappers.models_wrappers import ModelsWrapper
from models.classifiers_nlp import ClassifierNLP, ClassifierNLPRandomSearch
from utils.metrics import metric_f1_score


class ModelsWrapperNLP(ModelsWrapper):
    def __init__(
            self,
            models,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy', 'f1_score': make_scorer(metric_f1_score)},
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 3000},
            svd_kws={'n_components': 100}
    ):

        super().__init__(models=models,
                         tuner=tuner,
                         tuner_scoring=tuner_scoring,
                         final_scoring=final_scoring
                         )

        self.tfidf_kws = tfidf_kws
        self.svd_kws = svd_kws

    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierNLP(model, param_grid, tuner=self.tuner, tuner_scoring=self.tuner_scoring,
                            final_scoring=self.final_scoring, tfidf_kws=self.tfidf_kws)
        return clf.fit(X, y).cv_score(X, y)


class ModelsWrapperNLPRandomSearch(ModelsWrapperNLP):
    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierNLPRandomSearch(model, param_grid, tuner_scoring=self.tuner_scoring,
                                 final_scoring=self.final_scoring, tfidf_kws=self.tfidf_kws)
        return clf.fit(X, y).cv_score(X, y)