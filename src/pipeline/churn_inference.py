from pathlib import Path
import shutil 
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.model.evaluate_churn import churn_prediction, evaluate_model, evaluate_impact
from src.retention.explain  import get_top_drivers
from src.retention.persona  import assign_persona
from src.retention.retention import recommend_action

from src.features.processor import process_features
from src.utils.common import get_project_root
from src.utils.logger import logging

TARGET      = "Churned"
MODEL_NAME  = "churn_model"
ALIAS       = "best"

project_root = get_project_root()
serving_path = project_root / "artifacts" / "serving_model"

def _load_native_model(model_uri, model_type):

    if model_type == "xgboost":
        return mlflow.xgboost.load_model(model_uri)

    elif model_type == "lightgbm":
        return mlflow.lightgbm.load_model(model_uri)

    elif model_type == "catboost":
        return mlflow.catboost.load_model(model_uri)

    else:
        return mlflow.sklearn.load_model(model_uri)
    
def _get_model():
    """
    First use MLflow Model Registry to get 'best' model, Fallback to local artifacts if MLflow is unavailable.
    """
    
    model, model_type = None, None
    
    # connect mlflow tracking db
    mlflow_dir = project_root / "mlflow"
    tracking_db = f"sqlite:///{(mlflow_dir / 'tracking.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_db)
    
    try:
        """Get 'best' model from MLflow model registry."""
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run = client.get_run(model_version.run_id)
        model_type = run.data.tags.get("model_type", "unknown")
        model_uri_registry = f"models:/{MODEL_NAME}@{ALIAS}"

        logging.info("INFERENCE_MODEL: Syncing '%s' (%s) from MLflow Registry...", MODEL_NAME, model_type)
        
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri_registry)
        
        if serving_path.exists():
            shutil.rmtree(serving_path)
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_path, serving_path)
        
        # CRITICAL: Persist model type so future instances know which data to load
        with open(serving_path / "model_type.txt", "w") as f:
            f.write(model_type)

        model = _load_native_model(model_uri_registry, model_type)
        
    except Exception as e:
        """fallback to local artifacts for model"""
        logging.warning("INFERENCE_MODEL: MLflow Registry unavailable, attempting local fallback. Error: %s", e)
        if serving_path.exists():
            try:
                # Recover model_type from the sidecar file
                type_file = serving_path / "model_type.txt"
                if type_file.exists():
                    with open(type_file, "r") as f:
                        model_type = f.read().strip()
                model = _load_native_model(str(serving_path), model_type)
                
                logging.info("INFERENCE_MODEL: Loaded local model type: %s", model_type)
            except Exception as le:
                logging.error("INFERENCE_MODEL: Local model load failed: %s", le)
                return
        else:
            logging.error("INFERENCE_MODEL: No local model found at %s", serving_path)
            return
    
    finally:
        return model, model_type
        
    
def _build_dataset(data, model_type):
    """
    Transform the user provided data using the unified feature pipeline.
    """
    is_catboost = (model_type == "catboost")
    df_transformed = process_features(data, fit=False, is_catboost=is_catboost)
    
    # Extract target if present in the provided dataframe
    y_test = df_transformed[TARGET] if TARGET in df_transformed.columns else None
    
    # Prepare features for model prediction
    X_test = df_transformed.drop(columns=[TARGET]) if y_test is not None else df_transformed

    return X_test, y_test

def _prepare_inference_data(
    data: pd.DataFrame | None,
    n_samples: int | None,
    model,
    model_type
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series, pd.Series]:
    """
    Returns: X_test, y_test, y_prob, y_pred, segments
    y_test is None in inference-only mode (no labels)
    """
    # load data
    if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
            
        if model_type == "catboost":
            processed_test_file = project_root / "data" / "interim" / "transformed" / "test.parquet"
        else:
            processed_test_file = project_root / "data" / "processed" / "test.parquet"
        
        if not processed_test_file.exists():
            logging.error("INFERENCE_DATA: Test file not found at %s", processed_test_file)
            raise FileNotFoundError(f"Inference data file not found at [{processed_test_file}]")
            return {"Inference data not found": {processed_test_file}}

        df_inference_ready = pd.read_parquet(processed_test_file)
        logging.info(f"INFERENCE_DATA: load system test data from [{processed_test_file}]")
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples)
            logging.info("INFERENCE_DATA: Sampled %d rows from test set", n_samples)
            
        X_test = df_inference_ready.drop(columns=[TARGET])
        y_test = df_inference_ready[TARGET]
    
    else:
        X_test, y_test = _build_dataset(data, model_type)
        logging.info(f"INFERENCE_DATA: prepared user provided data for inference")
        
    y_prob, y_pred, segments = churn_prediction(X_test, model)
        
    return X_test, y_test, y_prob, y_pred, segments
    

def model_inference(
    data: pd.DataFrame | pd.Series | None = None, 
    n_samples: int = None, 
    output: str = "full"  # "full" | "metrics"
) -> dict:
    """
    Load the 'best' churn model and perform inference on the test set.
    Use model_type (CatBoost vs Others) to select the correct test data source.
    
    Return: 
    if output="full"; predictions per customer + metrics/summary
    if output="metrics"; aggregated metrics/summary only
    """
    
    model, model_type = _get_model()
    
    X_test, y_test, y_prob, y_pred, segments = _prepare_inference_data(data, n_samples, model, model_type)
        
    # per-customer predictions (always computed, conditionally returned)
    predictions_df = pd.DataFrame({
        "churn_probability": y_prob.round(4),
        "predicted_churn":   y_pred,
        "segment":           segments
    })
    
    # -------------------------------- inference-only mode (no labels) ------------------------------- #
    if y_test is None:
        summary = (
            predictions_df
            .groupby("segment")
            .agg(
                total=("predicted_churn", "count"),
                predicted_churners=("predicted_churn", "sum"),
                avg_churn_probability=("churn_probability", "mean")
            )
            .round(4)
            .to_dict(orient="index")
        )
        result = {"summary_by_segment": summary}
        
        if output == "full":
            result["predictions"] = predictions_df.to_dict(orient="records")
            if data is None:
                result['data'] = X_test.to_dict(orient="records")
            
        return result

     # ---------------------------------- label mode (y_test present) --------------------------------- #
    metrics = evaluate_model(X_test, y_test, y_prob, y_pred, segments)
    result = {"metrics": metrics}
    if output == "full":
        predictions_df["actual_churn"] = y_test.values
        result["predictions"] = predictions_df.to_dict(orient="records")
        if data is None:
                result['data'] = X_test.to_dict(orient="records")
                result['label'] = y_test.to_dict(orient="records")

    logging.info("MODEL_INFERENCE: evaluation complete, output=%s", output)
    return result


def business_inference(
    data: pd.DataFrame | None = None,
    n_samples: int | None = None,
    output: str = "full"
) -> dict:
    """
    Business impact metrics per segment.
    Requires labeled data — returns error if y_test unavailable.
    
    output="full"    → per-segment impact + per-customer predictions
    output="metrics" → per-segment impact only
    """

    model, model_type = _get_model()
    X_test, y_test, y_prob, y_pred, segments = _prepare_inference_data(
        data, n_samples, model, model_type
    )

    if y_test is None:
        logging.warning("BUSINESS_INFERENCE: No labels available, impact metrics require ground truth")
        return {"error": "Business impact metrics require labeled data"}
    
    try:
        impact = evaluate_impact(X_test, y_test, y_prob, y_pred, segments)
        result = {"business_impact": impact}
    except Exception as e:
        logging.warning("INFERENCE_IMPACT: Business impact evaluation skipped or failed: %s", e)
        return {"error": e}
    
    if output == "full":
        result["predictions"] = pd.DataFrame({
            "churn_probability": y_prob.round(4),
            "predicted_churn":   y_pred,
            "segment":           segments,
            "actual_churn":      y_test.values
        }).to_dict(orient="records")

    logging.info("BUSINESS_INFERENCE: evaluation complete, output=%s", output)
    return result

def retention_inference(
    data: pd.DataFrame | None = None,
    n_samples: int | None = None,
    output: str = "full"
) -> dict:
    """
    Per-churner SHAP based probability contribution, persona, and retention strategy.
    Only runs on predicted churners — no output mode needed, always per-customer.
    
    Return:
    if output="full"; retention summary + per-churner details
    if output="metrics"; retention summary only

    """

    model, model_type = _get_model()
    X_test, y_test, y_prob, y_pred, segments = _prepare_inference_data(
        data, n_samples, model, model_type
    )

    # filter to predicted churners only
    churner_mask = y_pred == 1
    X_churners   = X_test[churner_mask]
    prob_churners = y_prob[churner_mask]
    seg_churners  = segments[churner_mask]

    if X_churners.empty:
        logging.info("RETENTION_INFERENCE: No predicted churners in dataset")
        result = {"retention_summary": {"total_churners": 0}}
        if output == "full":
            result["churners"] = []
        return result

    shap_results = get_top_drivers(model, X_churners)

    results = []
    for i, (idx, _) in enumerate(X_churners.iterrows()):
        seg     = seg_churners[idx]
        drivers = shap_results[i]
        persona = assign_persona(drivers["all_churn_drivers"])
        action  = recommend_action(seg, persona["primary_persona"])

        results.append({
            "churn_probability":     round(float(prob_churners[idx]), 4),
            "segment":               seg,
            "top_churn_drivers":     drivers["top_churn_drivers"],
            "top_retention_signals": drivers["top_retention_signals"],
            "persona":               persona["primary_persona"],
            "secondary_persona":     persona["secondary_persona"],
            "retention_strategy":    action
        })

    # Aggregate summary metrics
    res_df = pd.DataFrame(results)
    
    # Flatten nested strategy fields for aggregation
    strategy_series = res_df["retention_strategy"]
    
    summary = {
        "total_churners": len(res_df),
        "avg_probability": round(float(res_df["churn_probability"].mean()), 4),
        "by_persona": res_df["persona"].value_counts().to_dict(),
        "by_segment": res_df["segment"].value_counts().to_dict(),
        "by_priority": strategy_series.apply(lambda x: x["priority"]).value_counts().to_dict(),
        "by_action": strategy_series.apply(lambda x: x["recommended_actions"]).explode().value_counts().to_dict()
    }

    result = {"retention_summary": summary}
    if output == "full":
        result["churners"] = results
        
    logging.info("RETENTION_INFERENCE: processed %d predicted churners, output=%s", len(results), output)
    return result


if __name__ == "__main__":
    print("building model inference")
    metrics = model_inference(output="metrics")
    print("model inference metrics: ", metrics)
    
    print("\nexecuting business inference")
    metrics = business_inference(output="metrics")
    print("business inference metrics: ", metrics)
    
    print("\nexecuting retention strategy")
    metrics = retention_inference(output="metrics")
    print("retention inference metrics: ", metrics)
