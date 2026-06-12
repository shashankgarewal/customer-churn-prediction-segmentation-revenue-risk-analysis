# Evaluate churn model performance module
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix
)

from src.utils.config import TARGET, RETENTION_RATES, THRESHOLDS


def _assign_segment(ltv):
    if ltv >= THRESHOLDS['VIP'][0]: return 'VIP'
    if ltv >= THRESHOLDS['IMP'][0]: return 'IMP'
    if ltv >= THRESHOLDS['High'][0]: return 'High'
    if ltv >= THRESHOLDS['Medium'][0]: return 'Medium'
    if ltv > THRESHOLDS['Low'][0]: return 'Low'
    return 'No_VALUE'

def churn_prediction(X_test, model):
    """
    return ltv segment, probability, and threshold based classification
    """
    y_prob = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    segments = X_test['Lifetime_Value'].apply(_assign_segment)
    
    segment_thresholds = pd.Series({seg: THRESHOLDS[seg][1] for seg in THRESHOLDS})
    row_thresholds = segments.map(segment_thresholds)

    y_pred = (y_prob >= row_thresholds).astype(int)
    
    return y_prob, y_pred, segments

def track_model(model, X_test, y_test) -> dict:
    """Evaluate churn model performance using metrics that are independent of threshold on unseen data
    Used for model comparision, tracking, and promption using mlflow"""
    
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

def evaluate_model(X_test, y_test, y_prob, y_pred, segments) -> dict:
    """
    Evaluate production model performance on classification ML metrics that align with business
    """

    df_eval = X_test.copy()
    df_eval[TARGET] = y_test.values
    df_eval['Segment'] = segments.values
    df_eval['Prob'] = y_prob.values
    df_eval['Pred'] = y_pred.values

    overall = {
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    per_segment = {}
    for seg in THRESHOLDS:
        seg_data = df_eval[df_eval['Segment'] == seg]
        if seg_data.empty:
            continue

        y_true_seg = seg_data[TARGET]
        y_pred_seg = seg_data['Pred']

        cm = confusion_matrix(y_true_seg, y_pred_seg, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        per_segment[seg] = {
            "threshold":  THRESHOLDS[seg][1],
            "total":      len(seg_data),
            "precision":  round(precision_score(y_true_seg, y_pred_seg, zero_division=0), 4),
            "recall":     round(recall_score(y_true_seg, y_pred_seg, zero_division=0), 4),
            "f1":         round(f1_score(y_true_seg, y_pred_seg, zero_division=0), 4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn)
        }

    return {"overall": overall, "per_segment": per_segment}

def evaluate_impact(X_test, y_test, y_prob, y_pred, segments) -> dict:
    """
    Evaluate business impact of churn prediction based customer intervention.
    """    

    df_eval = X_test.copy()
    df_eval[TARGET]    = y_test.values
    df_eval['Prob']    = y_prob.values
    df_eval['Pred']    = y_pred.values
    df_eval['Segment'] = segments.values

    per_segment = {}

    for seg in THRESHOLDS:
        seg_df = df_eval[df_eval['Segment'] == seg]
        if seg_df.empty:
            continue

        tp_df = seg_df[(seg_df['Pred'] == 1) & (seg_df[TARGET] == 1)]
        fn_df = seg_df[(seg_df['Pred'] == 0) & (seg_df[TARGET] == 1)]
        all_predicted_churners = seg_df[seg_df['Pred'] == 1]

        total_churn_exposure    = round(float(seg_df[seg_df[TARGET] == 1]['Lifetime_Value'].sum()), 2)
        revenue_at_risk         = round(float(tp_df['Lifetime_Value'].sum()), 2)
        missed_revenue          = round(float(fn_df['Lifetime_Value'].sum()), 2)
        retention_rate          = RETENTION_RATES.get(seg, 0.70)
        retention_savings       = round(revenue_at_risk * retention_rate, 2)
        
        expected_financial_risk = round(float(
            (all_predicted_churners['Lifetime_Value'] * all_predicted_churners['Prob']).sum()
        ), 2)
        
        churn_capture_rate      = round(
            revenue_at_risk / total_churn_exposure if total_churn_exposure > 0 else 0.0, 4
        )

        per_segment[seg] = {
            "total_churn_exposure":    total_churn_exposure,
            "revenue_at_risk":         revenue_at_risk,
            "retention_rate":          retention_rate,
            "retention_savings":       retention_savings,
            "missed_revenue":          missed_revenue,
            "expected_financial_risk": expected_financial_risk,
            "churn_capture_rate":      churn_capture_rate,
            "churners_identified":     int(len(tp_df)),
            "missed_churners":         int(len(fn_df))
        }

    return {"per_segment": per_segment}