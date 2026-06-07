import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, log_loss


def credit_risk_evaluation(y_true, y_pred_proba, model_name='Model'):
    auroc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auroc - 1
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = max(tpr - fpr)
    logloss = log_loss(y_true, y_pred_proba)

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(f"  AUROC    : {auroc:.4f}")
    print(f"  Gini     : {gini:.4f}")
    print(f"  KS Stat  : {ks:.4f}")
    print(f"  Log Loss : {logloss:.4f}")

    return {'AUROC': auroc, 'Gini': gini, 'KS': ks, 'LogLoss': logloss}


def calculate_psi(expected, actual, buckets=10):
    expected_pct = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_pct = np.histogram(
        actual, bins=np.histogram(expected, bins=buckets)[1]
    )[0] / len(actual)
    psi_values = (actual_pct - expected_pct) * np.log(
        (actual_pct + 1e-4) / (expected_pct + 1e-4)
    )
    psi = psi_values.sum()
    if psi < 0.10:
        status = 'Stable'
    elif psi < 0.25:
        status = 'Moderate shift - investigate'
    else:
        status = 'Major shift - recalibrate model'
    print(f"PSI: {round(psi, 4)} — {status}")
    return psi


def find_threshold_for_approval_rate(y_pred_proba, target_approval_rate=0.70):
    threshold = np.percentile(y_pred_proba, (1 - target_approval_rate) * 100)
    print(f"Threshold for {target_approval_rate*100}% approval rate: {round(threshold, 4)}")
    return threshold