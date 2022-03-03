import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_processing.stratified_sample_dataset import stratified_sample


def process_categorical(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].apply(lambda col: LabelEncoder().fit_transform(col))
    df[categorical_columns] = df[categorical_columns].astype(str)
    return df


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

    df = df.dropna().reset_index(drop=True)
    df = process_categorical(df)
    if nrows:
        df = stratified_sample(df, y_col, nrows)
    y = df[y_col]
    X = df.drop(columns=y_col)
    if isinstance(desired_categorical_columns, list):
        X[desired_categorical_columns] = X[desired_categorical_columns].astype(str)
    elif desired_categorical_columns == 'all':
        X = X.astype(str)
    _, y = np.unique(y, return_inverse=True)
    return X, y


def prepare_datasets_for_classification(datasets: dict, data_path='../data/'):
    result = []
    for dataset_name, (y_col, desired_categorical_columns, nrows) in datasets.items():
        X, y = prepare_dataset_for_classification(dataset_name, y_col, desired_categorical_columns, nrows, data_path)
        result.append(X)
        result.append(y)

    return result


# if __name__ == '__main__':
#     X, y = prepare_datasets_for_classification({'mushrooms.csv': ('class', 'all', None)})