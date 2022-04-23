import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set_style('whitegrid')


def visualize_results_on_boxplots_facet(
        results_path='../results/all_results.xlsx',
        out_path='../plots/results_boxplots.pdf',
        save=False,
        **kwargs
):

    df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name='accuracy')
    g = sns.catplot(data=melted, x='model', y='accuracy', col='dataset', kind='box', **kwargs)
    g.set_titles(template='{col_name}', size=12, fontweight='bold')
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    name = 'no_search'
    suffix = 'colab'
    visualize_results_on_boxplots_facet(f'../results_{suffix}/results_{name}.xlsx',
                                        f'../plots_{suffix}/results_{name}_boxplots_facet.pdf',
                                        palette=default_palette,
                                        col_wrap=4,
                                        sharey=False,
                                        save=False
    )



