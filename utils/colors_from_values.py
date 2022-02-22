import numpy as np
import pandas as pd
import seaborn as sns


def colors_from_values(values, palette_name, ascending=True, sort_by=None):
    values = values.sort_values(ascending=ascending).reset_index()
    if sort_by is not None:
        values[values.columns[0]] = pd.Categorical(values[values.columns[0]], sort_by, ordered=True)
    indices = values.sort_values(by=values.columns[0]).index
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)