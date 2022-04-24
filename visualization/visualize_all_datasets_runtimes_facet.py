import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set_style('whitegrid')


def visualize_runtimes_on_barplots_facet(
        results_path='../results/all_runtimes.xlsx',
        out_path='../plots/runtimes_barplots.pdf',
        df=None,
        time='runtime',
        minutes=False,
        save=False,
        **kwargs
):
    if df is None:
        df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name=time)
    g = sns.catplot(data=melted, x='model', y=time, col='dataset', kind='bar', ci=None,  **kwargs)
    g.set_titles(template='{col_name}', size=12, fontweight='bold')
    for ax in g.axes.flatten():
        if minutes:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y/60)}m'))
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y)}s'))
        for c in ax.containers:
            if minutes:
                labels = [f'{v.get_height()/60:.3f}m' for v in c]
            else:
                labels = [f'{v.get_height():.3f}s' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=10)
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    exp = 'TPE_tuning_100_25_trees'
    name = '12_datasets_TPE'
    in_path = f'../results/basic_metrics_runtimes/{exp}/tuning_times_{name}.xlsx'
    out_path = f'../plots/basic_metrics_runtimes/{exp}/tuning_times_{name}_facet.pdf'
    visualize_runtimes_on_barplots_facet(in_path,
                                         out_path,
                                         time='tuning time',
                                         minutes=True,
                                         palette=default_palette,
                                         col_wrap=4,
                                         sharey=False,
                                         save=True
    )
