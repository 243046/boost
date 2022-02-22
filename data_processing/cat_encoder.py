from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from category_encoders.cat_boost import CatBoostEncoder


class PermutedCatBoostEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _permute(self, X, y, seed=123):
        np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, y = X.iloc[perm, :].reset_index(drop=True), y[perm].reset_index(drop=True)
        return X, y

    def fit(self, X, y):
        X_, y_ = self._permute(X, y)
        self.encoder_ = CatBoostEncoder().fit(X_, y_)
        return self

    def transform(self, X, y=None):
        print(self.encoder_.transform(X))
        return self.encoder_.transform(X)


# class PermutedCatBoostEncoder(CatBoostEncoder):
#     def fit(self, X, y, **kwargs):
#         np.random.seed(123)
#         perm = np.random.permutation(len(X))
#         X, y = X.iloc[perm, :], y[perm]
#         super().fit(X, y, **kwargs)