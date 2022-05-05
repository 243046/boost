import pandas as pd


MODELS = ('GBM', 'XGBoost', 'LightGBM', 'CatBoost')
SCORINGS = ('accuracy', 'f1_score', 'AUC')
DATASETS = ['adult study', 'heart disease', 'amazon', 'mushrooms', 'breast cancer', 'churn', 'credit card fraud',
            'prostate', 'leukemia', 'gina agnostic', 'weather', 'IMDB reviews']


def get_table(
        df=None,
        drop=True
):

    df['dataset'] = df['dataset'].replace({
        'adult': 'adult study',
        'heart': 'heart disease',
        'creditcard': 'credit card fraud',
        'weather dataset': 'weather'

    })
    df = df[[df.columns[-1]] + df.columns[:-1].tolist()]
    mean_summary = df.groupby('dataset', as_index=False, sort=False).mean()
    sdev_summary = df.groupby('dataset', as_index=False, sort=False).std()
    if drop:
        return mean_summary.drop(columns='dataset'), sdev_summary.drop(columns='dataset')
    return mean_summary, sdev_summary


def add_plus(x):
    return f'+{x}' if x > 0 else x


def make_alternating_list(list1, list2):
    return [x for y in zip(list1, list2) for x in y]


def mean_sdev_summary(
        no_tuning_dir=r'../results/basic_metrics_runtimes/no_tuning_150_50_trees/',
        tpe_dir=fr'../results/basic_metrics_runtimes/TPE_tuning_150_50_trees/',
        rand_dir=fr'../results/basic_metrics_runtimes/randomized_30_tuning_150_50_trees/',
        no_vs_tpe=True,
        scoring=None
):
        print(f'{scoring = }')
        no_tuning_df = pd.read_excel(no_tuning_dir + f'results_{scoring}_12_datasets_no_tuning_150_50_trees.xlsx')
        no_tuning_mean, no_tuning_sdev = get_table(no_tuning_df)
        tpe_df = pd.read_excel(tpe_dir + f'results_{scoring}_12_datasets_TPE_150_50_trees.xlsx')
        tpe_mean, tpe_sdev = get_table(tpe_df)
        rand_df = pd.read_excel(rand_dir + f'results_{scoring}_12_datasets_randomized_30_150_50_trees.xlsx')
        rand_mean, rand_sdev = get_table(rand_df)

        # TPE vs no tuning
        df_mean = tpe_mean - no_tuning_mean
        df_mean_pct = (tpe_mean - no_tuning_mean) / no_tuning_mean * 100
        df_sdev = tpe_sdev - no_tuning_sdev
        df_sdev_pct = (tpe_sdev - no_tuning_sdev) / no_tuning_sdev * 100
        summary_mean = df_mean_pct.applymap(lambda x: fr'{add_plus(round(x, 3))}\%').astype(str).add_suffix(r'\\ mean')
        summary_sdev = df_sdev_pct.fillna(0).applymap(lambda x: fr'{add_plus(round(x, 3))}\%').astype(str).add_suffix(r'\\ sdev')
        summary = pd.concat([summary_mean, summary_sdev], axis=1)
        summary = summary[make_alternating_list(summary_mean.columns, summary_sdev.columns)]
        summary.insert(loc=0, column='dataset', value=DATASETS)
        #summary_mean.insert(loc=0, column='dataset', value=datasets)
        #summary_sdev.insert(loc=0, column='dataset', value=datasets)
        if no_vs_tpe:
            return summary


def make_table(
        df=None
):
    def print_rows(row):
        print(r' & '.join(row) + r'\\ \hline')

    def print_columns(cols):
        print(r'\textbf{dataset} &')
        for col in cols[1:-1]:
            print(r'\textbf{\begin{tabular}[c]{@{}c@{}}' + col + r'\end{tabular}} &')
        print(r'\textbf{\begin{tabular}[c]{@{}c@{}}' + cols[-1] + r'\end{tabular}} \\ \hline')

    def format_as_latex(df):
        print(r'\begin{landscape}')
        print(r'\begin{table}[h!]')
        print(r'\centering')
        print(r'\resizebox{400pt}{!}{')
        print(r'\begin{tabular}{|c|c|c|c|c|c|c|c|c|}')
        print(r'\hline')
        print_columns(df.columns)
        df.apply(print_rows, axis=1)
        print(r'\end{tabular}')
        print(r'}')
        print(r'\caption{}')
        print(r'\label{tab:}')
        print(r'\end{table}')
        print(r'\end{landscape}')

    format_as_latex(df)

if __name__ == '__main__':
    for scoring in SCORINGS:
        df = mean_sdev_summary(scoring=scoring)
        make_table(df)



