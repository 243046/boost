import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style('whitegrid')


def visualize_row_col_facet(
        results_path='../results/all_results.xlsx',
        out_path='../plots/results_boxplots.pdf',
        df=None,
        save=False,
        **kwargs
):
    if df is None:
        df = pd.read_excel(results_path)
    melted = df.melt(id_vars=['dataset', 'metric'], var_name='model', value_name='metric value')
    g = sns.catplot(data=melted, x='model', y='metric value', col='dataset', row='metric', kind='box', **kwargs)
    g.set_titles(size=12, fontweight='bold')
    if save:
        plt.savefig(out_path, bbox_inches='tight')
