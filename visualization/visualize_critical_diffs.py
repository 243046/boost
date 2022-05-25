import Orange
import matplotlib.pyplot as plt



acc = {'LightGBM\nrandomized': 9.75, 'LightGBM\nTPE': 7.54, 'XGBoost\nno tuning': 7.5, 'CatBoost\nno tuning': 7.12,
       'XGBoost\nrandomized': 6.88, 'GBM\nrandomized': 6.58, 'GBM\nTPE': 6.5, 'CatBoost\nTPE': 6.17,
       'CatBoost\nrandomized': 5.71, 'XGBoost\nTPE': 5.54, 'LightGBM\nno tuning': 5.25, 'GBM\nno tuning': 3.46}
f1 = {'LightGBM\nrandomized': 9.62, 'LightGBM\nTPE': 8.25, 'CatBoost\nno tuning': 7.54, 'XGBoost\nno tuning': 7.5,
      'XGBoost\nrandomized': 7.04, 'GBM\nrandomized': 6.62, 'GBM\nTPE': 6.17, 'CatBoost\nTPE': 6.08,
      'CatBoost\nrandomized': 5.5, 'XGBoost\nTPE': 5.33, 'LightGBM\nno tuning': 4.83, 'GBM\nno tuning': 3.5}
auc = {'LightGBM\nrandomized': 9.46, 'LightGBM\nTPE': 8.38, 'XGBoost\nrandomized': 7.21, 'XGBoost\nTPE': 6.88,
       'CatBoost\nno tuning': 6.75, 'XGBoost\nno tuning': 6.54, 'GBM\nTPE': 6.5, 'CatBoost\nTPE': 6.21,
       'GBM\nrandomized': 6.12, 'CatBoost\nrandomized': 5.96, 'LightGBM\nno tuning': 5.21, 'GBM\nno tuning': 2.79}



ranking = acc
names, avranks = list(ranking.keys()), list(ranking.values())
names = [x.replace('\n', ' ') for x in names]
cd = 4.810
print('CD value:', cd)
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=15, textspace=2.5, reverse=True)
plt.show()
plt.savefig('cd.pdf', bbox_inches='tight')


