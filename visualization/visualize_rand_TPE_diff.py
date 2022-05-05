import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set(font_scale=1.4)
sns.set_style('whitegrid')


def visualize_diffs_facet(
        tpe_path=r'../results/basic_metrics_runtimes/TPE_tuning_150_50_trees/tuning_times_12_datasets_TPE_150_50_trees.xlsx',
        rand_path=r'../results/basic_metrics_runtimes/randomized_30_tuning_150_50_trees/tuning_times_12_datasets_randomized_30_150_50_trees.xlsx',
        out_path='../plots/basic_metrics_runtimes/tuning_no_tuning_combined/diffs_barplots.pdf',
        time='tuning time',
        save=False,
        **kwargs
):

    df_tpe = pd.read_excel(tpe_path)
    df_tpe['dataset'] = df_tpe['dataset'].replace({
        'adult': 'adult study',
        'heart': 'heart disease',
        'creditcard': 'credit card fraud',
        'weather dataset': 'weather'
    })
    df_rand = pd.read_excel(rand_path)
    df_rand['dataset'] = df_rand['dataset'].replace({
        'adult': 'adult study',
        'heart': 'heart disease',
        'creditcard': 'credit card fraud',
        'weather dataset': 'weather'
    })
    df = (df_rand.drop(columns='dataset') - df_tpe.drop(columns='dataset')) / df_tpe.drop(columns='dataset') * 100
    df['dataset'] = df_rand['dataset']
    melted = df.melt(id_vars='dataset', var_name='model', value_name=time)
    g = sns.catplot(data=melted, x='model', y=time, col='dataset', kind='bar', ci=None,  **kwargs)
    g.set_titles(template='{col_name}', size=18, fontweight='bold')
    for ax in g.axes.flatten():
        formatter = lambda y, pos: f'+{int(y)}%' if y > 0 else f'{int(y)}%'
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        for c in ax.containers:
            labels = [f'+{v.get_height():.3f}%' if v.get_height() > 0 else f'{v.get_height():.3f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=14)
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    visualize_diffs_facet(palette=default_palette,
                          col_wrap=3,
                          sharey=False,
                          save=True
                          )
