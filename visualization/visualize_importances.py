import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')


def plot_importances(
        importances_path='../feature_importance/importances/importances.xlsx',
        out_path='../plots/importances.pdf',
        save=False,
        **kwargs
):

    df = pd.read_excel(importances_path)
    melted = df.melt(id_vars='model', var_name='feature', value_name='importance')
    g = sns.catplot(data=melted, x='importance', y='feature', orient='h', col='model', kind='bar', col_wrap=3,
                    sharex=False, sharey=False, **kwargs)
    g.set_titles(template='{col_name}')
    if save:
        plt.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    plot_importances('../feature_importance/importances/importances.xlsx',
                     '../plots/importances.pdf',
                     palette='viridis',
                     save=True
                     )
