import pandas as pd

from experiments.LightGBM_tuning.lightgbm_experiment import LightGBMExperiment


if __name__ == '__main__':
    model = LightGBMExperiment()
    model.fit()
    results = pd.DataFrame(model.cv_results_).drop(columns='params').filter(regex=r'(split)|(param)', axis=1)
    params_cols = results.filter(regex='param', axis=1).columns
    result = pd.DataFrame()
    for col in params_cols:
        temp = (
            results
            .filter(regex=fr'({col})|(split)', axis=1).dropna(subset=[col])
            .melt(id_vars=col, value_name='accuracy')
            .drop(columns='variable')
            .rename(columns={col: 'param value'})
        )
        temp['param'] = col.removeprefix('param_')
        result = pd.concat([result, temp])
    result.to_excel(r'../../results/LightGBM_tuning/results.xlsx', index=False)




