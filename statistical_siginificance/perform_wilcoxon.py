import pandas as pd
from scipy import stats


MODELS = ('GBM', 'XGBoost', 'LightGBM', 'CatBoost')
SCORINGS = ('accuracy', 'f1_score', 'AUC')


def get_mean_scores(df):
    return (
        df
        .groupby('dataset', as_index=False)
        .mean()
        .drop(columns='dataset')
    )


def compare_not_tuned_vs_tpe_randomized(
        no_tuning_dir=r'../results/basic_metrics_runtimes/no_tuning_150_50_trees/',
        tpe_dir=fr'../results/basic_metrics_runtimes/TPE_tuning_150_50_trees/',
        rand_dir=fr'../results/basic_metrics_runtimes/randomized_15_tuning_150_50_trees/'
):

    for model in MODELS:
        print(f'{model = }')
        for scoring in SCORINGS:
            print(f'{scoring = }')
            no_tuning_df = pd.read_excel(no_tuning_dir + f'results_{scoring}_12_datasets_no_tuning_150_50_trees.xlsx')
            no_tuning_df = get_mean_scores(no_tuning_df)
            tpe_df = pd.read_excel(tpe_dir + f'results_{scoring}_12_datasets_TPE_150_50_trees.xlsx')
            tpe_df = get_mean_scores(tpe_df)
            rand_df = pd.read_excel(rand_dir + f'results_{scoring}_12_datasets_randomized_15_150_50_trees.xlsx')
            rand_df = get_mean_scores(rand_df)

            # not tuned vs TPE
            model_agg_not_tuned, model_agg_tpe = no_tuning_df[model], tpe_df[model]
            stat, pvalue = stats.wilcoxon(model_agg_not_tuned, model_agg_tpe, alternative='less')
            if pvalue <= 0.05:
                print(f'TPE is better than not tuned, {pvalue = :.3f}')
            else:
                print(f'TPE and not tuned are equally good, {pvalue = :.3f}')

            # not tuned vs randomized
            model_agg_not_tuned, model_agg_rand = no_tuning_df[model], rand_df[model]
            stat, pvalue = stats.wilcoxon(model_agg_not_tuned, model_agg_rand, alternative='less')
            if pvalue <= 0.05:
                print(f'Randomized search is better than not tuned, {pvalue = :.3f}')
            else:
                print(f'Randomized search and not tuned are equally good, {pvalue = :.3f}')

            # TPE vs randomized
            model_agg_tpe, model_agg_rand = tpe_df[model], rand_df[model]
            stat, pvalue = stats.wilcoxon(model_agg_tpe, model_agg_rand, alternative='less')
            if pvalue <= 0.05:
                print(f'Randomized search is better than TPE, {pvalue = :.3f}\n')
            else:
                print(f'Randomized search and TPE are equally good, {pvalue = :.3f}\n')

        print('-'*20)


if __name__ =='__main__':
    compare_not_tuned_vs_tpe_randomized()