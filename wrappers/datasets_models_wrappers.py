import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score

from wrappers.models_wrappers import ModelsWrapper, ModelsWrapperRandomSearch
from utils.metrics import metric_f1_score

class DataModelsWrapper:
    def __init__(
            self,
            param_dict,
            tuner='hyperopt',
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')},
    ):

        self.param_dict = param_dict.copy()
        self.tuner = tuner
        self.tuner_scoring = tuner_scoring
        self.final_scoring = final_scoring

    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapper(models, tuner=self.tuner, tuner_scoring=self.tuner_scoring,
                         final_scoring=self.final_scoring)
        model.fit(X, y)
        return model.results_, model.tuning_times_, model.runtimes_

    @staticmethod
    def _get_mean_scores(results):
        return results.mean().to_frame().T

    def _fit(self):
        all_datasets_results = {scoring: pd.DataFrame() for scoring in self.final_scoring}
        all_datasets_tuning_times, all_datasets_runtimes = pd.DataFrame(), pd.DataFrame()

        for dataset_name, (X, y, models) in self.param_dict.items():
            print(dataset_name)
            results, tuning_times, runtimes = self._score_single_dataset(X, y, models)
            for scoring in results:
                results[scoring]['dataset'] = dataset_name
                all_datasets_results[scoring] = pd.concat([all_datasets_results[scoring], results[scoring]])
                all_datasets_results[scoring].to_excel(f'results_{scoring}.xlsx', index=False)
            tuning_times['dataset'] = dataset_name
            runtimes['dataset'] = dataset_name
            all_datasets_tuning_times = pd.concat([all_datasets_tuning_times, tuning_times])
            all_datasets_runtimes = pd.concat([all_datasets_runtimes, runtimes])
            all_datasets_tuning_times.to_excel(f'tuning_times.xlsx', index=False)
            all_datasets_runtimes.to_excel(f'runtimes.xlsx', index=False)
            print(f'dataset {dataset_name} done\n')

        self.all_datasets_results_ = all_datasets_results
        self.all_datasets_tuning_times_ = all_datasets_tuning_times
        self.all_datasets_runtimes_ = all_datasets_runtimes

    def fit(self):
        self._fit()
        return self


class DataModelsWrapperRandomSearch(DataModelsWrapper):
    def __init__(
            self,
            param_dict,
            tuner_scoring='neg_log_loss',
            final_scoring={'accuracy': 'accuracy',
                           'f1_score': make_scorer(metric_f1_score),
                           'AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')},
    ):

        super().__init__(param_dict=param_dict,
                         tuner=None,
                         tuner_scoring=tuner_scoring,
                         final_scoring=final_scoring
                         )

    def _score_single_dataset(self, X, y, models):
        model = ModelsWrapperRandomSearch(models, tuner_scoring=self.tuner_scoring, final_scoring=self.final_scoring)
        model.fit(X, y)
        return model.results_, model.tuning_times_, model.runtimes_
