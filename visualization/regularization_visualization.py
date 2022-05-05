import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_regularization(df, out_path='regularization_05_step.pdf', save=False):
    parameters = df.columns[:3]
    score = df.columns[-1]
    param_ranges = {
        param: (df[param].min(), df[param].max()) for param in parameters
    }
    score_min, score_max = df[score].min(), df[score].max()
    fig, ax = plt.subplots(3, 3, figsize=(12, 11))
    ax_index = 0
    for param in parameters:
        x_param, y_param = list(set(parameters) - {param})
        min_val, max_val = param_ranges[param]
        middle_val = (min_val + max_val) / 2
        for param_val in (min_val, middle_val, max_val):
            current_ax = ax.flatten()[ax_index]
            current_df = df[df[param] == param_val]
            pivot = current_df.pivot(index=x_param, columns=y_param, values=score)
            xticks, yticks = pivot.columns, pivot.index
            x_tick, y_tick = xticks[1] - xticks[0], yticks[1] - yticks[0]
            extent = [xticks.min() - x_tick/2, xticks.max() + x_tick/2, yticks.min() - y_tick/2, yticks.max() + y_tick/2]
            im = current_ax.imshow(pivot, interpolation='nearest', cmap='Reds', extent=extent,
                                   origin='lower', vmin=score_min, vmax=score_max, aspect='equal')
            current_ax.set_title(fr'$\{param}$ = {param_val}', fontsize=14)
            current_ax.set_xlabel(fr'$\{x_param}$', fontsize=12, labelpad=4)
            current_ax.set_ylabel(fr'$\{y_param}$', fontsize=12, labelpad=0)
            current_ax.set_xticks(xticks)
            current_ax.set_yticks(yticks)
            current_ax.tick_params(axis='both', which='major', labelsize=6)
            current_ax.tick_params(axis='both', which='minor', labelsize=6)
            current_ax.spines['top'].set_visible(False)
            current_ax.spines['right'].set_visible(False)
            current_ax.spines['bottom'].set_visible(False)
            current_ax.spines['left'].set_visible(False)
            ax_index += 1
    fig.subplots_adjust(right=0.82, hspace=.45, wspace=-.1)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(score, rotation=270, labelpad=15)
    if save:
        fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    paths = [
        (r'../results/xgboost_regularization/regularization_results_05_step_prostate.xlsx',
         r'../plots/xgboost_regularization/regularization_05_step_prostate.pdf'),
        (r'../results/xgboost_regularization/regularization_results_05_step_gina_agnostic.xlsx',
         r'../plots/xgboost_regularization/regularization_05_step_gina_agnostic.pdf'),
        (r'../results/xgboost_regularization/regularization_results_05_step_max_val_10_leukemia.xlsx',
         r'../plots/xgboost_regularization/regularization_05_step_leukemia.pdf')
    ]
    for path, out_path in paths:
        df = pd.read_excel(path)
        visualize_regularization(df, out_path=out_path, save=True)
