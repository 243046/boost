import pandas as pd

from data_processing.text_processor import TextProcessor


if __name__ == '__main__':
    df = pd.read_csv('../data/imdb_dataset_full.csv')
    X = df['review']
    processed = TextProcessor().fit_transform(X)
    df.insert(loc=1, column='review_cleared', value=processed)
    df.to_csv('../data/imdb_dataset_full.csv', index=False)

