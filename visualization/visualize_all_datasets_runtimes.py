import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')


def visualize_runtimes_on_barplots(
        results_path='../results/all_runtimes.xlsx',
        out_path='../plots/runtimes_barplots.pdf',
        figsize=(7, 12),
        save=False,
        **kwargs
):
    df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name='runtime')
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=melted, x='runtime', y='dataset', hue='model', ax=ax, **kwargs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}s'))
    for c in ax.containers:
        labels = [f'{v.get_width():.2f}s' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge', fontsize=10)
    if save:
        fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    visualize_runtimes_on_barplots('../results/all_datasets_runtimes.xlsx',
                                   '../plots/all_datasets_runtimes_barplots.pdf',
                                   palette='gnuplot',
                                   save=True
                                   )
