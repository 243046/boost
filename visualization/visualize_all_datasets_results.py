import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')


def visualize_results_on_boxplots(
        results_path='../results/all_results.xlsx',
        out_path='../plots/results_boxplots.pdf',
        figsize=(7, 12),
        save=False,
        **kwargs
):

    df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name='accuracy')
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=melted, x='accuracy', y='dataset', hue='model', ax=ax, **kwargs)
    if save:
        fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    visualize_results_on_boxplots('../results/results_no_search.xlsx',
                                  '../plots/results_no_search_boxplots.pdf',
                                  palette='Set2',
                                  save=True
                                  )
