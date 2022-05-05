import pandas as pd


def make_table(
        results_path='../results/all_results.xlsx',
        df=None
):
    def print_rows(row):
        print(r' & '.join(row) + r'\\ \hline')

    def format_as_latex(df):
        print(r'\begin{table}[h!]')
        print(r'\centering')
        print(r'\begin{tabular}{|c|c|c|c|c|}')
        print(r'\hline')
        print(r'\textbf{' + r'}  & \textbf{'.join(df.columns) + r'} \\ \hline')
        df.apply(print_rows, axis=1)
        print(r'\end{tabular}')
        print(r'\caption{}')
        print(r'\label{tab:}')
        print(r'\end{table}')

    if df is None:
        df = pd.read_excel(results_path)
    df['dataset'] = df['dataset'].replace({
        'adult': 'adult study',
        'heart': 'heart disease',
        'creditcard': 'credit card fraud',
        'weather dataset': 'weather'

    })
    df = df[[df.columns[-1]] + df.columns[:-1].tolist()]
    mean_summary = df.groupby('dataset', as_index=False, sort=False).mean()
    sdev_summary = df.groupby('dataset', as_index=False, sort=False).std()
    datasets = mean_summary['dataset']
    #df.applymap(lambda x: round(x, N - int(floor(log10(abs(x))))))

    summary = mean_summary.drop(columns='dataset').round(3).astype(str) + r' $\pm$ ' + \
              sdev_summary.drop(columns='dataset').round(3).astype(str)
    summary.insert(loc=0, column='dataset', value=datasets)
    print(format_as_latex(summary), '\n')


if __name__ == '__main__':
    combinations = [
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'accuracy'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'f1_score'),
        ('no_tuning_150_50_trees', '12_datasets_no_tuning_150_50_trees', 'AUC'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'accuracy'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'f1_score'),
        ('TPE_tuning_150_50_trees', '12_datasets_TPE_150_50_trees', 'AUC')
    ]
    for exp, name, scoring in combinations:
        print(exp, name, scoring)
        in_path = f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx'
        make_table(f'../results/basic_metrics_runtimes/{exp}/results_{scoring}_{name}.xlsx')



