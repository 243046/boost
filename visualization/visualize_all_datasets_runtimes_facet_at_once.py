import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set(font_scale=1.4)
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
    df['dataset'] = df['dataset'].replace({
        'adult': 'adult study',
        'heart': 'heart disease',
        'creditcard': 'credit card fraud',
        'weather dataset': 'weather'
    })
    melted = df.melt(id_vars='dataset', var_name='model', value_name=time)
    g = sns.catplot(data=melted, x='model', y=time, col='dataset', kind='bar', ci=None,  **kwargs)
    g.set_titles(template='{col_name}', size=18, fontweight='bold')
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
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=14)
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    combinations = [
        ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees'),
        ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees'),
        ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees'),
        ('randomized_30_tuning_150_50_trees', '12_datasets_randomized_30_150_50_trees')
    ]
    times = ('runtime', 'tuning_time')
    for time in times:
        minutes = False if time == 'runtime' else True
        for exp, name in combinations:
            in_path = f'../results/basic_metrics_runtimes/{exp}/{time}s_{name}.xlsx'
            out_path = f'../plots/basic_metrics_runtimes/{exp}/{time}s_{name}_facet.pdf'
            try:
                visualize_runtimes_on_barplots_facet(in_path,
                                                     out_path,
                                                     time=time.replace('_', ' '),
                                                     minutes=minutes,
                                                     palette=default_palette,
                                                     col_wrap=3,
                                                     sharey=False,
                                                     save=True
                )
            except:
                print('no tuning times found')
