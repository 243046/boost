import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set(font_scale=1.4)
sns.set_style('whitegrid')


def visualize_results_on_boxplots_facet(
        results_path='../results/all_results.xlsx',
        out_path='../plots/results_boxplots.pdf',
        df=None,
        metric='accuracy',
        save=False,
        **kwargs
):
    if df is None:
        df = pd.read_excel(results_path)
    df['dataset'] = df['dataset'].replace({
        'adult': 'adult study',
        'heart': 'heart disease',
        'creditcard': 'credit card fraud',
        'weather dataset': 'weather'
    })
    melted = df.melt(id_vars='dataset', var_name='model', value_name=metric)
    g = sns.catplot(data=melted, x='model', y=metric, col='dataset', kind='box', **kwargs)
    g.set_titles(template='{col_name}', size=18, fontweight='bold')
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    combinations = [
        ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees', 'accuracy'),
        ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees', 'f1_score'),
        ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees', 'accuracy'),
        ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees', 'f1_score'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'accuracy'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'f1_score'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'AUC'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'accuracy'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'f1_score'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'AUC'),
        ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees', 'accuracy'),
        ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees', 'f1_score'),
        ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees', 'AUC'),
        ('randomized_30_tuning_150_50_trees', '12_datasets_randomized_30_150_50_trees', 'accuracy'),
        ('randomized_30_tuning_150_50_trees', '12_datasets_randomized_30_150_50_trees', 'f1_score'),
        ('randomized_30_tuning_150_50_trees', '12_datasets_randomized_30_150_50_trees', 'AUC')
    ]
    for exp, name, scoring in combinations:
        in_path = f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx'
        out_path = f'../plots/basic_metrics_runtimes/{exp}/results_{scoring}_{name}_facet.pdf'
        visualize_results_on_boxplots_facet(in_path,
                                             out_path,
                                             metric=scoring.replace('f1_', 'F1 '),
                                             palette=default_palette,
                                             col_wrap=3,
                                             sharey=False,
                                             save=True
                                             )



