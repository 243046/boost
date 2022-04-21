import pandas as pd
import numpy as np


def stratified_sample(df, target, N):
    return (
           df
           .groupby(target, group_keys=False)
           .apply(lambda x: x.sample(int(np.rint(N * len(x) / len(df))), random_state=123))
           .sample(frac=1, random_state=123).reset_index(drop=True)
           )
