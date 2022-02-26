import pandas as pd

from wrappers.models_wrappers import ModelsWrapper, ModelsWrapperRandomSearch


class DataModelsWrapper(ModelsWrapper):
    def __init__(self, param_dict, tuner='hyperopt', scoring='accuracy'):
        self.param_dict = param_dict.copy()
        self.tuner = tuner
        self.scoring = scoring

    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapper(models, tuner=self.tuner, scoring=self.scoring)
        model.fit(X, y)
        return model.results_, model.runtimes_

    @staticmethod
    def _get_mean_scores(results):
        return results.mean().to_frame().T

    def _fit(self):
        all_datasets_results, all_datasets_runtimes, results_for_plotting, runtimes_for_plotting = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        datasets_names = list(self.param_dict.keys())

        for dataset_name, (X, y, models) in self.param_dict.items():
            results, runtimes = self._score_single_dataset(X, y, models)
            mean_scores = self._get_mean_scores(results)

            all_datasets_results = pd.concat([all_datasets_results, mean_scores])
            results['dataset'] = dataset_name
            all_datasets_runtimes = pd.concat([all_datasets_runtimes, runtimes])
            runtimes['dataset'] = dataset_name
            results_for_plotting = pd.concat([results_for_plotting, results])
            runtimes_for_plotting = pd.concat([runtimes_for_plotting, runtimes])

        all_datasets_results.index, all_datasets_runtimes.index = datasets_names, datasets_names
        self.all_datasets_results_ = all_datasets_results
        self.all_datasets_runtimes_ = all_datasets_runtimes
        self.results_for_plotting_ = results_for_plotting
        self.runtimes_for_plotting_ = runtimes_for_plotting

    def fit(self):
        self._fit()
        return self


class DataModelsWrapperRandomSearch(DataModelsWrapper):
    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperRandomSearch(models, scoring=self.scoring)
        model.fit(X, y)
        return model.results_, model.runtimes_
