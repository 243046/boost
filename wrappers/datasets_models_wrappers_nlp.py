from sklearn.metrics import make_scorer, roc_auc_score

from wrappers.datasets_models_wrappers import DataModelsWrapper
from wrappers.models_wrappers_nlp import ModelsWrapperNLP, ModelsWrapperNLPRandomSearch
from utils.metrics import metric_f1_score


class DataModelsWrapperNLP(DataModelsWrapper):
    def __init__(
            self,
            param_dict,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')},
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 3000},
            svd_kws={'n_components': 100}
    ):

        super().__init__(param_dict=param_dict,
                         tuner=tuner,
                         tuner_scoring=tuner_scoring,
                         final_scoring=final_scoring
                         )

        self.tfidf_kws = tfidf_kws
        self.svd_kws = svd_kws

    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperNLP(models, tuner=self.tuner, tuner_scoring=self.tuner_scoring,
                                 final_scoring=self.final_scoring, tfidf_kws=self.tfidf_kws)
        model.fit(X, y)
        return model.results_, model.tuning_times_, model.runtimes_


class DataModelsWrapperNLPRandomSearch(DataModelsWrapperNLP):
    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperNLPRandomSearch(models, tuner_scoring=self.tuner_scoring,
                                 final_scoring=self.final_scoring, tfidf_kws=self.tfidf_kws)
        model.fit(X, y)
        return model.results_, model.tuning_times_, model.runtimes_
