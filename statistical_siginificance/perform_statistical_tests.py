import itertools
import pandas as pd

from friedman_nemenyi import FriedmanNemenyi


def make_alternating_list(list1, list2):
    return [x for y in zip(list1, list2) for x in y]


def perform_all_combinations(combinations):
    for exp, name, scoring in combinations:
        print(exp, name, scoring)
        data_path = f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx'
        out_path = f'../plots/statistical_significance/{exp}/heatmap_{scoring}_{name}.pdf'
        c = FriedmanNemenyi(data_path, out_path, save=False)
        c.perform_analysis()


def create_combined_dfs(
        scorings=('accuracy', 'AUC', 'f1_score'),
        no_tuning_dir=r'../results/basic_metrics_runtimes/no_tuning_150_50_trees/',
        tuning_dir=r'../results/basic_metrics_runtimes/TPE_tuning_150_50_trees/',
        out_path=r'../results/basic_metrics_runtimes/tuning_no_tuning_combined/',
        variant='150_50_trees'
):

    for scoring in scorings:
        no_tuning_df = pd.read_excel(no_tuning_dir + f'results_{scoring}_12_datasets_no_tuning_{variant}.xlsx')
        tuning_df = pd.read_excel(tuning_dir + f'results_{scoring}_12_datasets_TPE_{variant}.xlsx')\
            .drop(columns='dataset').add_suffix('\ntuned')
        df = pd.concat([no_tuning_df, tuning_df], axis=1)
        new_cols = make_alternating_list(no_tuning_df.columns[:-1], tuning_df.columns) + ['dataset']
        df[new_cols].to_excel(out_path + f'results_{scoring}_12_datasets_{variant}.xlsx', index=False)


def perform_analysis_on_combined(scorings=('accuracy', 'AUC', 'f1_score'), variant='150_50_trees', save=False):
    for scoring in scorings:
        path = fr'../results/basic_metrics_runtimes/tuning_no_tuning_combined/results_{scoring}_12_datasets_{variant}.xlsx'
        out_path = fr'../plots/statistical_significance/tuning_no_tuning_combined/heatmap_{scoring}_{variant}.pdf'
        c = FriedmanNemenyi(path, out_path, save=save)
        c.perform_analysis()


if __name__ =='__main__':
    combinations = [
            # ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees', 'accuracy'),
            # ('no_tuning_100_25_trees', '12_datasets_no_tuning_100_25_trees', 'f1_score'),
            ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'accuracy'),
            ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'f1_score'),
            ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'AUC'),
            # ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees', 'accuracy'),
            # ('TPE_tuning_100_25_trees', '12_datasets_TPE_100_25_trees', 'f1_score'),
            ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'accuracy'),
            ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'f1_score'),
            ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'AUC')
        ]
    #perform_all_combinations(combinations)
    create_combined_dfs()
    perform_analysis_on_combined(save=True)