import numpy as np
import pandas as pd


def prepare_dataset_for_classification(
        dataset_name,
        y_col,
        desired_categorical_columns=[],
        nrows=None,
        data_path='../data/'
):

    path = data_path + dataset_name
    if 'xlsx' in dataset_name:
        df = pd.read_excel(path, nrows=nrows)
    else:
        df = pd.read_csv(path, nrows=nrows)

    df = df.dropna().convert_dtypes()
    y = df[y_col]
    X = df.drop(columns=y_col)
    if desired_categorical_columns:
        X[desired_categorical_columns] = X[desired_categorical_columns].astype(str)
    _, y = np.unique(y, return_inverse=True)
    return X, y


def prepare_datasets_for_classification(datasets: dict, data_path='../data/'):
    result = []
    for dataset_name, (y_col, desired_categorical_columns, nrows) in datasets.items():
        X, y = prepare_dataset_for_classification(dataset_name, y_col, desired_categorical_columns, nrows, data_path)
        result.append(X)
        result.append(y)

    return result


#a = prepare_datasets_for_classification({'mushrooms.csv': ('class', [], None)})