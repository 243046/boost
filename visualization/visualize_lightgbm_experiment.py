import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.palettes import default_palette

sns.set(font_scale=1.1)
sns.set_style('whitegrid')


def visualize_lightgbm_results_on_boxplots_facet(
        results_path=r'../results/LightGBM_tuning/results.csv',
        out_path=r'../plots/LightGBM_tuning/lightgbm_results_boxplots.pdf',
        save=False,
        **kwargs
):

    df = pd.read_csv(results_path, sep=r';')
    print(df)
    df['param value'] = df['param value'].round(2)
    g = sns.catplot(data=df, x='param value', y='accuracy', col='param', kind='box', **kwargs)
    g.set_titles(template='{col_name}', size=16, fontweight='bold')
    for i, ax in enumerate(g.axes.flatten()):
        if i == 1:
            ax.set_xticklabels(np.linspace(13, 130, 10).astype(int))
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    visualize_lightgbm_results_on_boxplots_facet(palette=[default_palette[2]],
                                                 col_wrap=2,
                                                 sharey=False,
                                                 sharex=False,
                                                 save=True
                                                 )



