"""
Export inference outputs as CSV files for Power BI dashboard.
Usage: python -m scripts.export_powerbi_data
"""
from pathlib import Path
import pandas as pd

from src.pipeline.churn_inference import model_inference, business_inference, retention_inference
from src.utils.common import get_project_root

project_root = get_project_root()
export_dir = project_root / "artifacts" / "powerbi"


def export_prediction_output():
    """Per-customer predictions with segment, LTV, probability, predicted and actual churn."""
    print("Running model inference...")
    result = model_inference(output="full")

    if "error" in result:
        raise RuntimeError(f"model_inference failed — {result['error']}")

    predictions = result.get("predictions", [])
    if not predictions:
        print("Warning: No predictions returned")
        return

    df = pd.DataFrame(predictions)
    if "customer_index" in df.columns:
        df = df.set_index("customer_index")
    path = export_dir / "predictions.csv"
    df.to_csv(path, index=True, index_label="customer_index")
    print(f"predictions.csv saved ({len(df)} rows) → {path}")


def export_retention_output():
    """Per-churner persona, SHAP drivers, priority, and recommended actions."""
    print("Running retention inference...")
    result = retention_inference(output="full")

    if "error" in result:
        raise RuntimeError(f"retention_inference failed — {result['error']}")

    churners = result.get("churners", [])
    if not churners:
        print("Warning: No churners returned")
        return

    rows = []
    for i, c in enumerate(churners):
        drivers = c.get("top_churn_drivers", [])
        actions = c.get("retention_strategy", {}).get("recommended_actions", [])
        rows.append({
            "customer_index":    c.get("customer_index", i),
            "segment":           c.get("segment"),
            "churn_probability": c.get("churn_probability"),
            "persona":           c.get("persona"),
            "secondary_persona": c.get("secondary_persona"),
            "priority":          c.get("retention_strategy", {}).get("priority"),
            "driver_1":          drivers[0]["feature"] if len(drivers) > 0 else None,
            "driver_2":          drivers[1]["feature"] if len(drivers) > 1 else None,
            "driver_3":          drivers[2]["feature"] if len(drivers) > 2 else None,
            "recommended_actions": " | ".join(actions),
        })

    df = pd.DataFrame(rows)
    path = export_dir / "retention.csv"
    df.to_csv(path, index=False)
    print(f"retention.csv saved ({len(df)} rows) -> {path}")


def export_business_impact():
    """Per-segment business impact — revenue at risk, savings, missed revenue."""
    print("Running business inference...")
    result = business_inference(output="metrics")

    if "error" in result:
        raise RuntimeError(f"business_inference failed — {result['error']}")

    per_seg = result.get("business_impact", {}).get("per_segment", {})
    if not per_seg:
        print("Warning: No business impact data returned")
        return

    rows = []
    for seg, v in per_seg.items():
        rows.append({
            "segment":              seg,
            "retention_rate":       v.get("retention_rate"),
            "churners_identified":  v.get("churners_identified"),
            "missed_churners":      v.get("missed_churners"),
            "churn_capture_rate":   v.get("churn_capture_rate"),
            "total_churn_exposure": v.get("total_churn_exposure"),
            "revenue_at_risk":      v.get("revenue_at_risk"),
            "retention_savings":    v.get("retention_savings"),
            "at_risk_not_saved":    round(v.get("revenue_at_risk", 0) - v.get("retention_savings", 0), 2),
            "missed_revenue":       v.get("missed_revenue"),
            "expected_financial_risk": v.get("expected_financial_risk"),
        })

    df = pd.DataFrame(rows)
    path = export_dir / "business_impact.csv"
    df.to_csv(path, index=False)
    print(f"business_impact.csv saved ({len(df)} rows) -> {path}")

if __name__ == "__main__":
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting Power BI data export to {export_dir}")

    export_prediction_output()
    export_retention_output()
    export_business_impact()

    print(f"Done. Files saved to {export_dir}")