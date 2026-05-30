# Script to reset the experiement and clean the associates logs and artifacts.
# Reset any experiment using --name={experiment_name} argument

import mlflow
import shutil, argparse
from pathlib import Path
from mlflow.entities import ViewType
from urllib.parse import urlparse

from src.utils.logger import logging
from src.utils.common import get_project_root

project_root = get_project_root()
serving_path = project_root / "artifacts" / "serving_model"
mlflow_dir = project_root / "mlflow"

tracking_db = f"sqlite:///{(mlflow_dir / 'tracking.db').as_posix()}"
mlflow.set_tracking_uri(tracking_db)

def delete_metadata(client: mlflow.tracking.MlflowClient, experiment_name: str = "churn-training-pipeline"):
    """Remove mlflow metadata from sql database safely."""
    exp = client.get_experiment_by_name(experiment_name)

    if exp:
        # Search all runs (Active and Deleted) to ensure metadata is purged
        runs = client.search_runs(experiment_ids=[exp.experiment_id], run_view_type=ViewType.ALL)
        for run in runs:
            if run.info.lifecycle_stage == 'active':
                client.delete_run(run.info.run_id)
                logging.info(f"RESET_EXP: delete run id {run.info.run_id}")
        
        try:
            # Delete the experiment record
            client.delete_experiment(exp.experiment_id)
            logging.info(f"RESET_EXP: remove full {experiment_name} experiment")
        except Exception as e:
            # Gracefully handle cases where the DB record is already marked as deleted
            logging.warning(f"RESET_EXP: Metadata deletion for '{experiment_name}' skipped: %s", e)
    else:
        logging.warning(f"RESET_EXP: '{experiment_name}' experiment not found")

def delete_artifacts(client: mlflow.tracking.MlflowClient, experiment_name: str = "churn-training-pipeline"):
    """Hard delete artifact files for a specific experiment's runs only."""
    exp = client.get_experiment_by_name(experiment_name)

    if not exp:
        logging.warning(f"RESET_EXP: '{experiment_name}' not found, skipping artifact deletion")
        return

    # Include all runs to clean up orphaned files from previously interrupted deletions
    runs = client.search_runs(experiment_ids=[exp.experiment_id], run_view_type=ViewType.ALL)
    
    for run in runs:
        if not run.info.artifact_uri:
            continue
            
        # Robustly parse the local path from the URI
        parsed_uri = urlparse(run.info.artifact_uri)
        path_str = parsed_uri.path
        # Handle Windows paths where urlparse might leave a leading slash (e.g., /C:/...)
        if path_str.startswith('/') and len(path_str) > 2 and path_str[2] == ':':
            path_str = path_str.lstrip('/')
        run_artifact_path = Path(path_str)
        parent = run_artifact_path.parent

        if parent.exists():
            shutil.rmtree(parent)
            logging.info(f"RESET_EXP: deleted run artifacts {parent}")
        else:
            logging.warning(f"RESET_EXP: run artifact path not found, skipping {parent}")

    run_ids = {run.info.run_id for run in runs}
    models_dir = mlflow_dir / "models"
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            mlmodel_file = model_dir / "artifacts" / "MLmodel"
            if mlmodel_file.exists():
                content = mlmodel_file.read_text()
                if any(run_id in content for run_id in run_ids):
                    shutil.rmtree(model_dir)
                    logging.info(f"RESET_EXP: deleted model artifacts {model_dir}")
                    
    # Clear the local serving directory to prevent inference fallback to old models
    if serving_path.exists():
        shutil.rmtree(serving_path)
        logging.info(f"RESET_EXP: cleared local serving model directory")
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset MLflow experiment and clean artifacts")
    parser.add_argument(
        "--name", 
        type=str, 
        default="churn-training-pipeline",
        help="Name of the MLflow experiment to delete"
    )
    args = parser.parse_args()

    # Create single client to avoid SQLite session conflicts
    mlflow_client = mlflow.tracking.MlflowClient()
    delete_artifacts(mlflow_client, args.name)
    delete_metadata(mlflow_client, args.name)
