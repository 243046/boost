import pandas as pd
import numpy as np


def stratified_sample(df, target, N):
    return (
           df
           .groupby(target, group_keys=False)
           .apply(lambda x: x.sample(int(np.rint(N * len(x) / len(df)))))
           .sample(frac=1).reset_index(drop=True)
           )


if __name__ == '__main__':
    df = pd.read_csv(f'../data/creditcard_full.csv')
    sampled = stratified_sample(df, 'Class', 8000)
    sampled.to_csv('../data/creditcard.csv', index=False)

    adult = pd.read_csv(f'../data/adult_full.csv')
    sampled = stratified_sample(adult, 'profit', 10000)
    sampled.to_csv('../data/adult.csv', index=False)
