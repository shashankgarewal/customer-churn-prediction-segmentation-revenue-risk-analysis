import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

TARGET = 'Churned'
RETENTION_RATE = 0.7 # out of 100 - 70 customer stays after intervention
THRESHOLDS = {
    'VIP': [4400, 0.18],
    'IMP': [2600, 0.2],
    'High': [1800, 0.4],
    'Medium': [800, 0.4],
    'Low': [0, 0.55],
}

def _assign_segment(ltv):
    if ltv >= THRESHOLDS['VIP'][0]: return 'VIP'
    if ltv >= THRESHOLDS['IMP'][0]: return 'IMP'
    if ltv >= THRESHOLDS['High'][0]: return 'High'
    if ltv >= THRESHOLDS['Medium'][0]: return 'Medium'
    if ltv > THRESHOLDS['Low'][0]: return 'Low'
    return 'No_VALUE'

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate churn model performance using metrics that are independent of threshold on unseen data"""
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ap": average_precision_score(y_test, y_prob),
        "roc": roc_auc_score(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
    }

    # Prepare dataframe for segment level metrics
    df_eval = X_test.copy()
    df_eval[TARGET] = y_test.values
    df_eval['Segment'] = df_eval['Lifetime_Value'].apply(_assign_segment)
    df_eval['Prob'] = y_prob
    
    for seg, threshold_vals in THRESHOLDS.items():
        prob_threshold = threshold_vals[1]
        seg_data = df_eval[df_eval['Segment'] == seg]
        
        if not seg_data.empty:
            y_true_seg = seg_data[TARGET]
            y_prob_seg = seg_data['Prob']
            # y_pred_seg = (y_prob_seg >= prob_threshold).astype(int)

            # metrics[f"recall_{seg.lower()}"] = recall_score(y_true_seg, y_pred_seg, zero_division=0)
            metrics[f"ap_{seg.lower()}"] = average_precision_score(y_true_seg, y_prob_seg)  if y_true_seg.sum() > 0 else 0.0

            # ROC-AUC requires both classes to be present in the segment slice
            if y_true_seg.nunique() > 1:
                metrics[f"roc_{seg.lower()}"] = roc_auc_score(y_true_seg, y_prob_seg)
            else:
                metrics[f"roc_{seg.lower()}"] = 0.5
        else:
            metrics[f"ap_{seg.lower()}"] = 0.0
            metrics[f"roc_{seg.lower()}"] = 0.5

    return metrics

def plot_segment_distributions(model, X, y) -> dict[str, plt.Figure]:
    """Generate probability distribution plots for each segment.
    Purpose: Determine probability threshold for each segment"""
    
    y_prob = model.predict_proba(X)[:, 1]
    df = X.copy()
    df[TARGET] = y.values
    df['Prob'] = y_prob
    df['Segment'] = df['Lifetime_Value'].apply(_assign_segment)

    figs = {}

    for seg in THRESHOLDS.keys():
        seg_data = df[df['Segment'] == seg]
        if not seg_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.kdeplot(data=seg_data, x='Prob', hue=TARGET, fill=True, ax=ax, common_norm=False)
            
            ax.set_title(f"Probability Distribution - {seg}")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            figs[seg] = fig

    return figs

def evaluate_impact(model, X, y) -> dict:
    """Evaluate business impact of churn prediction based customer intervention"""    
    # Implementation logic for impact remains...
    seg_metrics = dict()
    

    
    return seg_metrics