import pandas as pd
from sklearn.metrics import recall_score, average_precision_score, roc_auc_score

TARGET = 'Churned'
RETENTION_RATE = 0.7 # out of 100 - 70 customer stays after intervention
THRESHOLDS = {
    'VIP': [4400, 0.25],
    'IMP': [2600, 0.3],
    'High': [1800, 0.4],
    'Medium': [800, 0.5],
    'Low': [0, 0.65],
}

def _assign_segment(ltv):
    if ltv >= THRESHOLDS['VIP'][0]: return 'VIP'
    if ltv >= THRESHOLDS['IMP'][0]: return 'IMP'
    if ltv >= THRESHOLDS['High'][0]: return 'High'
    if ltv >= THRESHOLDS['Medium'][0]: return 'Medium'
    if ltv > THRESHOLDS['Low'][0]: return 'Low'
    return 'No_VALUE'

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate churn model performance of unseen data"""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "recall": recall_score(y_test, y_pred),
        "avg_precision": average_precision_score(y_test, y_prob),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }

    # segment level metrics
    df_eval = pd.concat([X_test, y_test], axis=1).copy()
    df_eval['Segment'] = df_eval['Lifetime_Value'].apply(_assign_segment)
    df_eval['Prob'] = y_prob
    
    for seg in THRESHOLDS.keys():
        seg_data = df_eval.query("Segment == @seg")
        if not seg_data.empty:
            df_eval['Prediction'] = int(df_eval['Prob'] >= THRESHOLDS[seg][1])
            
            recall = recall_score(seg_data[TARGET], seg_data['Prediction'], zero_division=0)
            avg_precision = average_precision_score(seg_data[TARGET], seg_data['Prediction'], zero_division=0)
            
            metrics[f"recall_{seg.lower()}"] = recall
            metrics[f"avg_precision_{seg.lower()}"] = avg_precision
            
    return metrics

def evaluate_impact(model, X) -> dict:
    """Evaluate business impact of churn prediction based customer intervention"""    
    
    X['Segment'] = X['Lifetime_Value'].apply(_assign_segment)
    X['Prob'] = y_prob

    def get_churn_prediction(churn_prob):
        if churn_prob >= THRESHOLDS['VIP'][1]: return 1
        if churn_prob >= THRESHOLDS['IMP'][1]: return 1
        if churn_prob >= THRESHOLDS['High'][1]: return 1
        if churn_prob >= THRESHOLDS['Medium'][1]: return 1
        if churn_prob > THRESHOLDS['Low'][1]: return 1
        return 0
    
    df_eval = pd.concat([X, y], axis=1).copy()
    df_eval['Segment'] = df_eval['Lifetime_Value'].apply(_assign_segment)
    df_eval['Churn_Pred'] = df_eval['Churn_prob'] = df_eval['Lifetime_Value'].apply(assign_segment)
    
    y_prob = model.predict_proba(X)[:, 1]
    seg_metrics = dict()
    

    
    return seg_metrics