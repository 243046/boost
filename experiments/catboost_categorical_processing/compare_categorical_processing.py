import warnings

from experiments.catboost_categorical_processing.categorical_classifier import CatBoostExperiment
from data_processing.process_dataset import prepare_datasets_for_classification
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
