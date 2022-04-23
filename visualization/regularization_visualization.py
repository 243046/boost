import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# def visualize_regularization(df, out_path='regularization_05_step_prostate.pdf'):
#     parameters = df.columns[:3]
#     score = df.columns[-1]
#     param_ranges = {
#         param: (df[param].min(), df[param].max()) for param in parameters
#     }
#     fig, ax = plt.subplots(3, 3, figsize=(12, 11))
#     ax_index = 0
#     for param in parameters:
#         x_param, y_param = list(set(parameters) - {param})
#         min_val, max_val = param_ranges[param]
#         middle_val = (min_val + max_val) / 2
#         for param_val in (min_val, middle_val, max_val):
#             current_df = df[df[param] == param_val]
#             pivot_table = current_df.pivot(index=x_param, columns=y_param, values=score)
#             g = sns.heatmap(pivot_table, ax=ax.flatten()[ax_index], cmap='Reds', xticklabels=True, yticklabels=True)
#             g.invert_yaxis()
#             g.axes.set_title(fr'$\{param}$ = {param_val}', fontsize=10)
#             g.axes.set_xlabel(fr'$\{x_param}$', labelpad=8)
#             g.axes.set_ylabel(fr'$\{y_param}$')
#             #g.axes.set_yticklabels(g.axes.get_yticklabels(), rotation=90, ha='right')
#             g.set_xticklabels(g.get_xticklabels(), fontsize=8)
#             g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
#             ax_index += 1
#     plt.tight_layout()
#     #plt.savefig(out_path, bbox_inches='tight')


def visualize_regularization(df, out_path='regularization_05_step.pdf'):
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
            extent = [xticks.min(), xticks.max(), yticks.min(), yticks.max()]
            im = current_ax.imshow(pivot, interpolation='nearest', cmap='Reds', extent=extent,
                                   origin='lower', vmin=score_min, vmax=score_max, aspect='equal')
            current_ax.set_title(fr'$\{param}$ = {param_val}', fontsize=10)
            current_ax.set_xlabel(fr'$\{x_param}$', labelpad=4)
            current_ax.set_ylabel(fr'$\{y_param}$', labelpad=0)
            current_ax.set_xticks(xticks)
            current_ax.set_yticks(yticks)
            current_ax.tick_params(axis='both', which='major', labelsize=6)
            current_ax.tick_params(axis='both', which='minor', labelsize=6)
            current_ax.spines['top'].set_visible(False)
            current_ax.spines['right'].set_visible(False)
            current_ax.spines['bottom'].set_visible(False)
            current_ax.spines['left'].set_visible(False)
            ax_index += 1
    fig.subplots_adjust(right=0.82, hspace=.37, wspace=-.1)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(score, rotation=270, labelpad=15)
    fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    name = 'gina_agnostic'
    df = pd.read_excel(f'regularization_results_05_step_{name}.xlsx')
    visualize_regularization(df, out_path=f'regularization_05_step_{name}.pdf')
