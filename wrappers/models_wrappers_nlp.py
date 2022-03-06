from wrappers.models_wrappers import ModelsWrapper
from models.classifiers_nlp import ClassifierNLP, ClassifierNLPRandomSearch


class ModelsWrapperNLP(ModelsWrapper):
    def __init__(
            self,
            models,
            tuner='hyperopt',
            scoring='accuracy',
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000},
            svd_kws={'n_components': 100}
    ):

        super().__init__(models=models,
                         tuner=tuner,
                         scoring=scoring
                         )

        self.tfidf_kws = tfidf_kws
        self.svd_kws = svd_kws

    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierNLP(model, param_grid, tuner=self.tuner, scoring=self.scoring, tfidf_kws=self.tfidf_kws)
        clf.fit(X, y)
        return clf.cv_score(X, y)


class ModelsWrapperNLPRandomSearch(ModelsWrapperNLP):
    def _score_single_model(self, X, y, model, param_grid):
        clf = ClassifierNLPRandomSearch(model, param_grid, scoring=self.scoring, tfidf_kws=self.tfidf_kws)
        clf.fit(X, y)
        return clf.cv_score(X, y)