from wrappers.datasets_models_wrappers import DataModelsWrapper, DataModelsWrapperRandomSearch
from wrappers.datasets_models_wrappers_nlp import DataModelsWrapperNLP, DataModelsWrapperNLPRandomSearch


def run(param_dict, mode='randomized', tuner='hyperopt'):
    if mode == 'randomized':
        model = DataModelsWrapperRandomSearch(param_dict)
    elif mode == 'TPE':
        model = DataModelsWrapper(param_dict, tuner=tuner)
    model.fit()
    return model.all_datasets_results_, model.all_datasets_tuning_times_, model.all_datasets_runtimes_


def run_nlp(param_dict, mode='randomized', tuner='hyperopt',
            tfidf_kws={'ngram_range': (1, 2), 'min_df': 3, 'max_features': 10000}):
    if mode == 'randomized':
        model = DataModelsWrapperNLPRandomSearch(param_dict, tfidf_kws=tfidf_kws)
    elif mode == 'TPE':
        model = DataModelsWrapperNLP(param_dict, tuner=tuner, tfidf_kws=tfidf_kws)
    model.fit()
    return model.all_datasets_results_, model.all_datasets_tuning_times_, model.all_datasets_runtimes_