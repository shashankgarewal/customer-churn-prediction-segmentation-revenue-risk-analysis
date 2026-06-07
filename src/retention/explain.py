## SHAP - top n contributing features for all churners at local levelh
import shap
import pandas as pd

EXCLUDE_FROM_DRIVERS = ["Lifetime_Value"]

def get_top_drivers(model, X: pd.DataFrame, top_n: int = 3) -> list[dict]:
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # compatibility with old SHAP that produces list with 2 arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ignore ltv as this is part of retention strategy for ltv segments
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df = shap_df.drop(columns=EXCLUDE_FROM_DRIVERS, errors="ignore")

    results = []

    for _, row in shap_df.iterrows():
        
        churn_drivers = (
            row[row > 0]
            .nlargest(top_n)
            .round(4)
            .reset_index()
            .rename(columns={'index': 'feature', 0: 'impact'})
            .to_dict(orient='records')
        )
        
        retention_signals = (
            row[row < 0]
            .abs()
            .nlargest(top_n)
            .round(4)
            .reset_index()
            .rename(columns={'index': 'feature', 0: 'impact'})
            .to_dict(orient='records')
        )
            
        results.append({

            "top_churn_drivers": churn_drivers,

            "top_retention_signals": retention_signals,

            "all_churn_drivers": (
                row[row > 0]
                .round(4)
                .to_dict()
            ),

            "all_retention_signals": (
                row[row < 0]
                .abs()
                .round(4)
                .to_dict()
            )
        })
        
    return results