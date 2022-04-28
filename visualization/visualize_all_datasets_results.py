import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set_style('whitegrid')


def visualize_results_on_boxplots(
        results_path='../results/all_results.xlsx',
        out_path='../plots/results_boxplots.pdf',
        df=None,
        metric='accuracy',
        figsize=(7, 12),
        save=False,
        **kwargs
):

    if df is None:
        df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name=metric)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=melted, x=metric, y='dataset', hue='model', ax=ax, **kwargs)
    if save:
        fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    name = 'high_dimensional'
    suffix = 'colab'
    visualize_results_on_boxplots(f'../results_{suffix}/results_{name}.xlsx',
                                  f'../plots_{suffix}/results_{name}_boxplots.pdf',
                                  palette=default_palette,
                                  save=True
                                  )
