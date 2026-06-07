import shap
import matplotlib.pyplot as plt
import numpy as np


def get_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_sample, save_path=None):
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.title('Global Feature Importance — SHAP Values')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def explain_individual(explainer, shap_values, X_sample, idx=0, save_path=None):
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[idx],
        feature_names=X_sample.columns.tolist()
    ))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def get_top_risk_drivers(shap_values, feature_names, n=5):
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = mean_shap.argsort()[::-1][:n]
    drivers = [(feature_names[i], round(mean_shap[i], 4)) for i in top_idx]
    print(f"\nTop {n} Risk Drivers:")
    for name, val in drivers:
        print(f"  {name}: {val}")
    return drivers