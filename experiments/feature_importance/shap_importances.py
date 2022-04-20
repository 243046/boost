import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def sigmoid(x):
    return 1/(1+np.exp(-x))


def generate_dataset(n_features=5):
    np.random.seed(123)
    X = np.random.normal(size=(1000, n_features))
    features = [f'X_{i}' for i in range(n_features)]
    X = pd.DataFrame(X, columns=features)
    beta = 3**(np.arange(X.shape[1], 0, -1) -1)
    y = 1*(sigmoid(X @ beta) > .5)
    return X, y


def generate_shap_values(X, y, models):
    shap_dict = {}
    for model_name, model in models.items():
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_dict[model_name] = shap_values[1] if model_name == 'LightGBM' else shap_values
    return shap_dict


def visualize_shap(X, shap_dict, nrow=2, ncol=2):
    for i, model_name in enumerate(shap_dict):
        plt.subplot(nrow, ncol, i+1)
        shap.summary_plot(shap_dict[model_name], X, color_bar=False, plot_size=[14, 10])
        plt.title(model_name, fontsize=12)
        plt.xlabel('SHAP value', fontsize=12)
        plt.ylabel('')
    plt.subplots_adjust(hspace=.3)
    plt.savefig('shap_values.pdf', bbox_inches='tight')


if __name__ == '__main__':
    models = {
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(n_estimators=100, verbose=False, random_state=123)
    }
    X, y = generate_dataset()
    shap_dict = generate_shap_values(X, y, models)
    visualize_shap(X, shap_dict)