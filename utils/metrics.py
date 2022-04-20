import numpy as np
from sklearn.metrics import f1_score


def metric_f1_score(y_true, y_pred):
    if (len(np.unique(y_true)) == 2) and (len(np.unique(y_pred)) == 2):
        return f1_score(y_true, y_pred, average='binary')
    return f1_score(y_true, y_pred, average='weighted')
