import pandas as pd
import numpy as np

from data_processing.stratified_sample_dataset import stratified_sample

def prepare_nlp_for_classification(
        dataset_name,
        text_column,
        y_col,
        nrows=None,
        data_path='../../data/'
):

    path = data_path + dataset_name
    if 'xlsx' in dataset_name:
        df = pd.read_excel(path, nrows=nrows)
    else:
        df = pd.read_csv(path, nrows=nrows)

    df = df.dropna().reset_index(drop=True)
    if nrows:
        df = stratified_sample(df, y_col, nrows)
    y = df[y_col]
    X = df[text_column]
    _, y = np.unique(y, return_inverse=True)
    return X, y


# if __name__ == '__main__':
#     X, y = prepare_nlp_for_classification({'mushrooms.csv': ('class', 'all', None)})