import warnings
import pandas as pd
import seaborn as sns

from visualization.visualize_all_datasets_runtimes import visualize_runtimes_on_barplots
from visualization.visualize_results_row_col_facet import visualize_row_col_facet

sns.set(font_scale=1.2)
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    rename_dict = {
        'different\nperm.': 'CatBoost\nEncoder',
        'embedded\nidentical perm.': 'embedded\nOrdered',
        'embedded\nPlain': 'embedded\nPlain'
    }
    results = pd.read_excel(r'../results/catboost_categorical_processing/results_all_metrics.xlsx').rename(columns=rename_dict)
    runtimes = pd.read_excel(r'../results/catboost_categorical_processing/runtimes.xlsx').rename(columns=rename_dict)

    visualize_runtimes_on_barplots(df=runtimes,
                                   palette='hot',
                                   out_path=fr'../plots/catboost_categorical_processing/catboost_runtimes.pdf',
                                   save=True
                                   )
    visualize_row_col_facet(df=results,
                            palette='hot',
                            out_path=fr'../plots/catboost_categorical_processing/catboost_results_all_metrics.pdf',
                            sharey=False,
                            save=True
                            )
