import warnings
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from experiments.catboost_categorical_processing.categorical_classifier import CatBoostExperiment
from data_processing.process_dataset import prepare_datasets_for_classification
from visualization.visualize_all_datasets_results import visualize_results_on_boxplots
from visualization.visualize_all_datasets_runtimes import visualize_runtimes_on_barplots
from visualization.visualize_results_row_col_facet import visualize_row_col_facet
from visualization.palettes import default_palette
warnings.filterwarnings('ignore')


def run_categorical(X, y, dataset_name):
    model = CatBoostExperiment(dataset_name=dataset_name)
    model.fit(X, y)
    return model.results_, model.runtimes_


if __name__ == '__main__':
    d = {
        'mushrooms.csv': ('class', 'all', None),
        'amazon.csv': ('ACTION', 'all', None)
    }

    X_1, y_1, X_2, y_2 = prepare_datasets_for_classification(d, data_path='../../data/')

    results_1, runtimes_1 = run_categorical(X_1, y_1, 'mushrooms')
    results_2, runtimes_2 = run_categorical(X_2, y_2, 'amazon')
    results = pd.DataFrame()
    for scoring in results_1:
        df_path = fr'../../results/catboost_categorical_processing/results_{scoring}.xlsx'
        out_path = fr'../../plots/catboost_categorical_processing/results_{scoring}.pdf'
        df = pd.concat([results_1[scoring], results_2[scoring]])
        df.to_excel(df_path, index=False)
        visualize_results_on_boxplots(df=df,
                                      metric=scoring,
                                      palette='cool',
                                      out_path=out_path,
                                      save=True
                                      )
        df['metric'] = scoring
        results = pd.concat([results, df])

    results['metric'] = results['metric'].str.replace('f1_', 'F1 ')
    results.to_excel(r'../../results/catboost_categorical_processing/results_all_metrics.xlsx', index=False)
    runtimes = pd.concat([runtimes_1, runtimes_2])
    runtimes.to_excel(r'../../results/catboost_categorical_processing/runtimes.xlsx', index=False)
    visualize_runtimes_on_barplots(df=runtimes,
                                   palette='cool',
                                   out_path=fr'../../plots/catboost_categorical_processing/runtimes.pdf',
                                   save=True
                                   )
    visualize_row_col_facet(df=results,
                            palette='cool',
                            out_path=fr'../../plots/catboost_categorical_processing/results_all_metrics.pdf',
                            sharey=False,
                            save=True
                            )
