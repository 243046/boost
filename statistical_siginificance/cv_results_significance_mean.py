import pandas as pd
from scipy import stats
import itertools

name = 'no_search'
suffix = '_colab'
cv_results = pd.read_excel(f'../results{suffix}/results_{name}.xlsx')
datasets = cv_results['dataset'].unique()
models = cv_results.drop(columns='dataset').columns
models_pairs = list(itertools.combinations(models, 2))

for dataset in datasets:
    print(dataset)
    current_results = cv_results[cv_results['dataset'] == dataset]
    for model_1, model_2 in models_pairs:
        sample_1, sample_2 = current_results[model_1], current_results[model_2]
        pval = stats.ttest_ind(sample_1, sample_2, equal_var=True, alternative='greater', random_state=123).pvalue
        if pval <= 0.05:  # reject null hypothesis -> means are different
            print(f'{model_1} is better than {model_2}, p = {pval:.3f}')
        else:
            print(f'{model_1} and {model_2} are equally good, p = {pval:.3f}')
    print(dataset, 'done\n')