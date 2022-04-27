import os
import pandas as pd
from visualization.visualize_all_datasets_results_facet import visualize_results_on_boxplots_facet
from visualization.visualize_all_datasets_runtimes_facet import visualize_runtimes_on_barplots_facet


results_path = r'C:/Users/p/Documents/M/2 stopień/masters thesis/boost/results/basic_metrics_runtimes/'
plots_path = r'C:/Users/p/Documents/M/2 stopień/masters thesis/boost/plots/basic_metrics_runtimes/'
experiments_folders = [x[0] for x in os.walk(results_path)]
for folder in experiments_folders:
    print(folder)
    experiment_name = folder.split('\\')[1]
    if not os.path.exists(plots_path + experiment_name):
        os.mkdir(plots_path + experiment_name)
    files_in_folder = os.listdir(folder)
    for file in files_in_folder:
        print(file)
        # if 'results' in file:
        #     visualize_results_on_boxplots_facet(file,
        #                                         plots_path + experiment_name,
        #                                         palette=default_palette,
        #                                         col_wrap=4,
        #                                         sharey=False,
        #                                         save=False
        #                                         )
