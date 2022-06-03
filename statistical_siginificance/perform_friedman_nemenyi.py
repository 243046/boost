import itertools
import pandas as pd

from friedman_nemenyi import FriedmanNemenyi


def make_alternating_list(list1, list2):
    return [x for y in zip(list1, list2) for x in y]


def make_alternating_list_3(list1, list2, list3):
    return [item for sublist in zip(list1, list2, list3) for item in sublist]


def perform_all_combinations(combinations, save=False):
    for exp, name, scoring in combinations:
        print(exp, name, scoring)
        data_path = f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx'
        out_path = f'../plots/statistical_significance/{exp}/heatmap_{scoring}_{name}.pdf'
        c = FriedmanNemenyi(data_path, out_path, save=save)
        c.perform_analysis()


def create_combined_dfs(
        scorings=('accuracy', 'AUC', 'f1_score'),
        no_tuning_dir=r'../results/basic_metrics_runtimes/no_tuning_150_50_trees/',
        out_path=r'../results/basic_metrics_runtimes/tuning_no_tuning_combined/',
        variant='150_50_trees',
        tuning='TPE'
):

    tuning_dir = fr'../results/basic_metrics_runtimes/{tuning}_tuning_150_50_trees/'
    for scoring in scorings:
        no_tuning_df = pd.read_excel(no_tuning_dir + f'results_{scoring}_12_datasets_no_tuning_{variant}.xlsx')
        tuning_df = pd.read_excel(tuning_dir + f'results_{scoring}_12_datasets_{tuning}_{variant}.xlsx')\
            .drop(columns='dataset').add_suffix('\ntuned')
        df = pd.concat([no_tuning_df, tuning_df], axis=1)
        new_cols = make_alternating_list(no_tuning_df.columns[:-1], tuning_df.columns) + ['dataset']
        df[new_cols].to_excel(out_path + f'results_{scoring}_12_datasets_{tuning}_{variant}.xlsx', index=False)


def perform_analysis_on_combined(scorings=('accuracy', 'AUC', 'f1_score'), variant='150_50_trees',
                                 tuning='TPE', save=False):
    for scoring in scorings:
        path = fr'../results/basic_metrics_runtimes/tuning_no_tuning_combined/results_{scoring}_12_datasets_{tuning}_{variant}.xlsx'
        out_path = fr'../plots/statistical_significance/tuning_no_tuning_combined/heatmap_{scoring}_{tuning}_{variant}.pdf'
        print(scoring)
        c = FriedmanNemenyi(path, out_path, save=save)
        c.perform_analysis(fontsize=6)


def create_combined_dfs_3(
        scorings=('accuracy', 'AUC', 'f1_score'),
        no_tuning_dir=r'../results/basic_metrics_runtimes/no_tuning_150_50_trees/',
        tpe_dir=r'../results/basic_metrics_runtimes/TPE_tuning_150_50_trees/',
        rand_dir=r'../results/basic_metrics_runtimes/randomized_30_tuning_150_50_trees/',
        out_path=r'../results/basic_metrics_runtimes/tuning_no_tuning_combined/',
        variant='150_50_trees'
):

    for scoring in scorings:
        no_tuning_df = pd.read_excel(no_tuning_dir + f'results_{scoring}_12_datasets_no_tuning_{variant}.xlsx')
        datasets = no_tuning_df['dataset']
        no_tuning_df = no_tuning_df.drop(columns='dataset').add_suffix('\nno tuning')
        tpe_df = pd.read_excel(tpe_dir + f'results_{scoring}_12_datasets_TPE_{variant}.xlsx')\
            .drop(columns='dataset').add_suffix('\nTPE')
        rand_df = pd.read_excel(rand_dir + f'results_{scoring}_12_datasets_randomized_30_{variant}.xlsx') \
            .drop(columns='dataset').add_suffix('\nrandomized')
        df = pd.concat([no_tuning_df, tpe_df, rand_df], axis=1)
        df['dataset'] = datasets
        new_cols = make_alternating_list_3(no_tuning_df.columns, tpe_df.columns, rand_df.columns) + ['dataset']
        df[new_cols].to_excel(out_path + f'results_{scoring}_12_datasets_all_3_{variant}.xlsx', index=False)


def perform_analysis_on_combined_3(scorings=('accuracy', 'f1_score', 'AUC'), variant='150_50_trees'):
    for scoring in scorings:
        path = fr'../results/basic_metrics_runtimes/tuning_no_tuning_combined/results_{scoring}_12_datasets_all_3_{variant}.xlsx'
        print(scoring)
        c = FriedmanNemenyi(path, '', save=False)
        c.perform_analysis(fontsize=6)


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
            ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'AUC'),
            ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees', 'accuracy'),
            ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees', 'f1_score'),
            ('randomized_15_tuning_150_50_trees', '12_datasets_randomized_15_150_50_trees', 'AUC')
        ]
    # perform_all_combinations(combinations, save=False)
    #create_combined_dfs(tuning='randomized_30')
    # perform_analysis_on_combined(tuning='randomized_15', save=False)
    # create_combined_dfs()
    # perform_analysis_on_combined(save=False)
    # data_path = f'../results/basic_metrics_runtimes/no_tuning_150_50_trees/nice.xlsx'
    # out_path = f'../plots/statistical_significance/heatmap.pdf'
    # c = FriedmanNemenyi(data_path, out_path, save=False)
    # c.perform_analysis()
    # create_combined_dfs_3()
    perform_analysis_on_combined_3()