from pathlib import Path
import shutil 
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.common import get_project_root
from src.model.evaluate_churn import evaluate_model, evaluate_impact
from src.utils.logger import logging

MODEL_NAME = "churn_model"
ALIAS = "best"

project_root = get_project_root()
serving_path = project_root / "artifacts" / "serving_model"

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

        logging.info("GET_MODEL: Syncing '%s' (%s) from MLflow Registry...", MODEL_NAME, model_type)
        
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri_registry)
        
        if serving_path.exists():
            shutil.rmtree(serving_path)
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_path, serving_path)
        
        # CRITICAL: Persist model type so future instances know which data to load
        with open(serving_path / "model_type.txt", "w") as f:
            f.write(model_type)

        model = mlflow.pyfunc.load_model(model_uri_registry)
        
    except Exception as e:
        """fallback to local artifacts for model"""
        logging.warning("GET_MODEL: MLflow Registry unavailable, attempting local fallback. Error: %s", e)
        if serving_path.exists():
            try:
                model = mlflow.pyfunc.load_model(str(serving_path))
                # Recover model_type from the sidecar file
                type_file = serving_path / "model_type.txt"
                if type_file.exists():
                    with open(type_file, "r") as f:
                        model_type = f.read().strip()
                logging.info("GET_MODEL: Loaded local model type: %s", model_type)
            except Exception as le:
                logging.error("GET_MODEL: Local model load failed: %s", le)
                return
        else:
            logging.error("GET_MODEL: No local model found at %s", serving_path)
            return
    
    finally:
        return model, model_type
        
    
def _build_dataset(X_test, model_type):
    """Transform the user provided data - placeholder for api and ui interaction"""
    y_test = pd.DataFrame()
    
    return X_test, y_test
    
def run_inference(X_test: pd.DataFrame | None = None, y_test: pd.DataFrame | None = None) -> None:
    """
    Load the 'best' churn model and perform inference on the test set.
    Use model_type (CatBoost vs Others) to select the correct test data source.
    """
    
    model, model_type = _get_model()
    
    
    if X_test:
            
        if model_type == "catboost":
            processed_test_file = project_root / "data" / "interim" / "imputed" / "test.parquet"
        else:
            processed_test_file = project_root / "data" / "processed" / "test.parquet"
        
        if not processed_test_file.exists():
            logging.error("INFERENCE_DATA_ERROR: Test file not found at %s", processed_test_file)
            return

        df_inference_ready = pd.read_parquet(processed_test_file)
        target = "Churned"
        
        X_test = df_inference_ready.drop(columns=[target])
        y_test = df_inference_ready[target]
    
    else:
        X_test, y_test = _build_dataset(X_test, model_type)

    evaluate_model(model, X_test, y_test)

    try:
        evaluate_impact(model, X_test)
    except Exception as e:
        logging.warning("INFERENCE_IMPACT: Business impact evaluation skipped or failed: %s", e)

if __name__ == "__main__":
    run_inference()
