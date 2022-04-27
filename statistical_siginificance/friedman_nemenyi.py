import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt

import stac


class FriedmanNemenyi:
    def __init__(self, data_path, out_path, alpha=0.05, save=False):
        self.data_path = data_path
        self.out_path = out_path
        self.alpha = alpha
        self.save = save

    def _load_data(self):
        df = pd.read_excel(self.data_path)
        self.data = (
            df
            .groupby('dataset', as_index=False)
            .mean()
            .drop(columns='dataset')
            .to_numpy()
        )
        self.models = df.columns[:-1].tolist()

    @staticmethod
    def _generate_scores(method, method_args, data, labels):
        pairwise_scores = method(data, **method_args)
        # pairwise_scores = pd.DataFrame([
        #     [1, 0.0001, 0.01, 0.5],
        #     [0.0001, 1, 0.2, 0.002],
        #     [0.01, 0.2, 1, 0.1],
        #     [0.5, 0.002, 0.1, 1]
        # ])
        pairwise_scores.set_axis(labels, axis='columns', inplace=True)
        pairwise_scores.set_axis(labels, axis='rows', inplace=True)
        return pairwise_scores

    def _plot_heatmap(
            self,
            scores,
            ranks,
    ):

        plt.figure()
        heatmap_args = {
            'flat': False,
            'linewidths': 0.25,
            'linecolor': '0.5',
            'square': True,
            'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3],
            'ranks': ranks
        }
        sp.sign_plot(scores, **heatmap_args)
        if self.save:
            plt.savefig(self.out_path, bbox_inches='tight')

    @staticmethod
    def _prepare_ranks_for_heatmap(ranks):
        new_ranks = np.vstack([list(ranks.values()) for _ in ranks]).round(2).astype(str)
        np.fill_diagonal(new_ranks, '')
        ranks_v = ('   \n' + pd.DataFrame(new_ranks)).values
        ranks_h = (pd.DataFrame(new_ranks.T) + '\n   ').values
        return ranks_v, ranks_h

    def _test_friedman(self):
        stat, p = stats.friedmanchisquare(*self.data)
        reject = p <= self.alpha
        # print(f'Should we reject H0 at the {(1 - self.alpha) * 100}% confidence level? {reject}, {p = }')
        return reject

    def _get_friedman_ranks(self):
        _, _, ranking, _ = stac.friedman_test(*self.data.T)
        ranks = dict(zip(self.models, ranking))
        print({k: round(v, 3) for k, v in sorted(ranks.items(), key=lambda x: x[1])}, '\n')
        return ranks

    def _test_nemenyi(self, reject):
        if not reject:
            raise Exception(
                'Exiting early. The rankings are only relevant if there was a difference'
                ' in the means i.e. if we rejected H0 above')
        nemenyi_scores = self._generate_scores(sp.posthoc_nemenyi_friedman, {}, self.data, self.models)
        return nemenyi_scores

    def perform_analysis(self):
        self._load_data()
        reject = self._test_friedman()
        ranks = self._get_friedman_ranks()
        nemenyi_scores = self._test_nemenyi(reject)
        heatmap_ranks = self._prepare_ranks_for_heatmap(ranks)
        self._plot_heatmap(nemenyi_scores, heatmap_ranks)


if __name__ == '__main__':
    combinations = [
        # ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees', 'accuracy'),
        # ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees', 'f1_score'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'accuracy'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'f1_score'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'AUC'),
        # ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees', 'accuracy'),
        # ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees', 'f1_score'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'accuracy'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'f1_score'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'AUC')
    ]
    for exp, name, scoring in combinations:
        print(exp, name, scoring)
        data_path = f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx'
        out_path = f'../plots/statistical_significance/{exp}/heatmap_{scoring}_{name}.pdf'
        c = FriedmanNemenyi(data_path, out_path, save=False)
        c.perform_analysis()
