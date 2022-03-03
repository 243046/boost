from wrappers.datasets_models_wrappers import DataModelsWrapper
from wrappers.models_wrappers_nlp import ModelsWrapperNLP, ModelsWrapperNLPRandomSearch


class DataModelsWrapperNLP(DataModelsWrapper):
    def __init__(
            self,
            param_dict,
            tuner='hyperopt',
            scoring='accuracy',
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}
    ):

        super().__init__(param_dict=param_dict,
                         tuner=tuner,
                         scoring=scoring
                         )

        self.tfidf_kws = tfidf_kws

    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperNLP(models, tuner=self.tuner, scoring=self.scoring, tfidf_kws=self.tfidf_kws)
        model.fit(X, y)
        return model.results_, model.runtimes_


class DataModelsWrapperNLPRandomSearch(DataModelsWrapperNLP):
    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperNLPRandomSearch(models, scoring=self.scoring, tfidf_kws=self.tfidf_kws)
        model.fit(X, y)
        return model.results_, model.runtimes_
