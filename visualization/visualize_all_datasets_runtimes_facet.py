import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set_style('whitegrid')


def visualize_runtimes_on_barplots_facet(
        results_path='../results/all_runtimes.xlsx',
        out_path='../plots/runtimes_barplots.pdf',
        save=False,
        **kwargs
):
    df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name='runtime')
    g = sns.catplot(data=melted, x='model', y='runtime', col='dataset', kind='bar', ci=None,  **kwargs)
    g.set_titles(template='{col_name}', size=12, fontweight='bold')
    for ax in g.axes.flatten():
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y)}s'))
        for c in ax.containers:
            labels = [f'{v.get_height():.3f}s' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=10)
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    name = 'no_search'
    suffix = 'colab'
    visualize_runtimes_on_barplots_facet(f'../results_{suffix}/runtimes_{name}.xlsx',
                                         f'../plots_{suffix}/runtimes_{name}_barplots_facet.pdf',
                                         palette=default_palette,
                                         col_wrap=4,
                                         sharey=False,
                                         save=False
    )
