import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set_style('whitegrid')


def visualize_results_on_boxplots_facet(
        results_path='../results/all_results.xlsx',
        out_path='../plots/results_boxplots.pdf',
        df=None,
        save=False,
        **kwargs
):
    if df is None:
        df = pd.read_excel(results_path)
    melted = df.melt(id_vars='dataset', var_name='model', value_name='accuracy')
    g = sns.catplot(data=melted, x='model', y='accuracy', col='dataset', kind='box', **kwargs)
    g.set_titles(template='{col_name}', size=12, fontweight='bold')
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    exp = 'no_tuning_150_50_trees'
    name = '12_datasets_no_tuning_150_50_trees'  # no_tuning TPE
    scoring = 'accuracy'
    # results_accuracy_12_datasets_no_tuning.xlsx
    # runtimes_12_datasets_no_tuning.xlsx
    in_path = f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx'
    out_path = f'../plots/basic_metrics_runtimes/{exp}/results_{scoring}_{name}_facet.pdf'
    visualize_results_on_boxplots_facet(in_path,
                                         out_path,
                                         palette=default_palette,
                                         col_wrap=4,
                                         sharey=False,
                                         save=True
                                         )



