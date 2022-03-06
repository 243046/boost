from time import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from visualization.palettes import default_palette

sns.set_style('whitegrid')



def runtimes_features(models, n_features_list, n_samples=100):
    cols = models.keys()
    results = []
    for n_features in n_features_list:
        X, y = make_classification(n_samples=n_samples, n_features=n_features)
        record = []
        for model in models.copy().values():
            t0 = time()
            model.fit(X, y)
            t = time()
            record.append(t-t0)
        results.append(record)
    results = pd.DataFrame(results, columns=cols)
    results['n_features'] = n_features_list
    return results


def runtimes_samples(models, n_samples_list, n_features=10):
    cols = models.keys()
    results = []
    for n_samples in n_samples_list:
        X, y = make_classification(n_samples=n_samples, n_features=n_features)
        record = []
        for model in models.copy().values():
            t0 = time()
            model.fit(X, y)
            t = time()
            record.append(t-t0)
        results.append(record)
    results = pd.DataFrame(results, columns=cols)
    results['n_samples'] = n_samples_list
    return results


def visualize_results(
        results_features,
        results_samples,
        out_path='../plots/runtimes_features_samples.pdf',
        figsize=(12, 8),
        base=10,
        save=False,
        **kwargs
):

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    melted_features = results_features.melt(id_vars='n_features', value_name='runtime', var_name='model')
    sns.lineplot(data=melted_features, x='n_features', y='runtime', hue='model', marker='o', ax=ax[0], **kwargs)
    ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}s'))
    ax[0].set_xscale('log', base=base)
    ax[0].set_xlabel(r'$n_{features}$')

    melted_samples = results_samples.melt(id_vars='n_samples', value_name='runtime', var_name='model')
    sns.lineplot(data=melted_samples, x='n_samples', y='runtime', hue='model', marker='o', ax=ax[1], **kwargs)
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}s'))
    ax[1].set_xscale('log', base=base)
    ax[1].set_xlabel(r'$n_{samples}$')

    if save:
        fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    n_estimators = 100
    models = {
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=n_estimators),
            'XGBoost': XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=123),
            'LightGBM': LGBMClassifier(n_estimators=n_estimators),
            'CatBoost': CatBoostClassifier(n_estimators=n_estimators, verbose=False, random_state=123)
    }

    base = 5
    n_features_list = base**np.arange(1, 7)
    n_samples_list = base**np.arange(1, 9)

    results_features = runtimes_features(models, n_features_list)
    print('features done')
    results_samples = runtimes_samples(models, n_samples_list)
    print('samples done')
    visualize_results(results_features, results_samples, base=base, palette=default_palette, save=True)