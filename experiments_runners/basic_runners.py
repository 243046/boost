from sklearn.metrics import make_scorer, roc_auc_score

from wrappers.datasets_models_wrappers import DataModelsWrapper, DataModelsWrapperRandomSearch
from wrappers.datasets_models_wrappers_nlp import DataModelsWrapperNLP, DataModelsWrapperNLPRandomSearch
from utils.metrics import metric_f1_score

def run(param_dict,
        mode='randomized',
        tuner='hyperopt',
        final_scoring={'accuracy': 'accuracy',
                       'f1_score': make_scorer(metric_f1_score),
                       'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')}
    ):
    if mode == 'randomized':
        model = DataModelsWrapperRandomSearch(param_dict, final_scoring=final_scoring)
    elif mode == 'TPE':
        model = DataModelsWrapper(param_dict, tuner=tuner, final_scoring=final_scoring)
    model.fit()
    return model.all_datasets_results_, model.all_datasets_tuning_times_, model.all_datasets_runtimes_


def run_nlp(param_dict,
            mode='randomized',
            tuner='hyperopt',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')},
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}):
    if mode == 'randomized':
        model = DataModelsWrapperNLPRandomSearch(param_dict, tfidf_kws=tfidf_kws, final_scoring=final_scoring)
    elif mode == 'TPE':
        model = DataModelsWrapperNLP(param_dict, tuner=tuner, tfidf_kws=tfidf_kws, final_scoring=final_scoring)
    model.fit()
    return model.all_datasets_results_, model.all_datasets_tuning_times_, model.all_datasets_runtimes_