import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
)
import optuna
import mlflow

from src.utils.common import get_project_root
from src.utils.logger import logging

RANDOM_STATE = 42 


# -------------------------------------- hyperparmeter setup ------------------------------------- #
def _tune_model(model_name, X, y):
    """hyperparameter setup with optuna and cross-validation"""

    def objective(trial):
        match model_name:
            case "randomforest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1
                }
                model = RandomForestClassifier(**params)
            case "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "random_state": RANDOM_STATE
                }
                model = XGBClassifier(**params)

            case "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 300, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                    "random_seed": RANDOM_STATE,
                    "verbose": False,
                    "allow_writing_files": False
                }
                model = CatBoostClassifier(**params)

            case "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "max_depth": trial.suggest_int("max_depth", -1, 15),
                    "random_state": RANDOM_STATE
                }
                model = LGBMClassifier(**params)

            case "hgboost":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "max_iter": trial.suggest_int("max_iter", 200, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "random_state": RANDOM_STATE
                }
                model = HistGradientBoostingClassifier(**params)
            case _:
                raise ValueError(f"Unsupported model for tuning: {model_name}")

                
        scores = cross_val_score(model, X, y, cv=3, scoring="recall", n_jobs=-1)
        return scores.mean()
    
    return objective


# ------------------------------- evaluate model and log in mlflow ------------------------------- #
def _evaluate_model(model, X_test, y_test, y_pred):
    """Evaluate and log model metrics across ltv segments in mlflow"""
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    eval_df = X_test.copy()
    eval_df['target'] = y_test
    eval_df['prediction'] = y_pred
    
    # Static Thresholds from Notebook (A5 Policy)
    thresholds = {
        'VIP': 4400,
        'IMP': 2600,
        'High': 1800,
        'Medium': 800,
    }

    def assign_segment(ltv):
        if ltv >= thresholds['VIP']: return 'VIP'
        if ltv >= thresholds['IMP']: return 'IMP'
        if ltv >= thresholds['High']: return 'High'
        if ltv >= thresholds['Medium']: return 'Medium'
        return 'Low'

    eval_df['Segment'] = eval_df['Lifetime_Value'].apply(assign_segment)
    
    segment_metrics = {}
    for seg in ['VIP', 'IMP', 'High', 'Medium', 'Low']:
        seg_data = eval_df.query("Segment == @seg")
        if not seg_data.empty:
            recall = recall_score(seg_data['target'], seg_data['prediction'], zero_division=0)
            segment_metrics[f"recall_{seg.lower()}"] = recall

    mlflow.log_metrics(segment_metrics)
    return segment_metrics


def build_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    target: str = "Churned", 
    models: list[str] | None = None,
    n_trials: int = 15
):
    """Construct and track model tuning experiments."""
    experiment_name = "churn-training-pipeline"
    project_root = Path(get_project_root())
    
    mlflow_dir = project_root / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)
    
    artifact_root = (mlflow_dir / "experiment_logs").as_uri()
    tracking_db = f"sqlite:///{(mlflow_dir / 'tracking.db').as_posix()}"

    mlflow.set_tracking_uri(tracking_db)

    # set log_model and log_artifact location at create experiment
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root
        )

    mlflow.set_experiment(experiment_name)
    
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]
    
    if not models:
        models = ["randomforest", "xgboost", "hgboost", "catboost", "lightgbm"]
    
    for model_name in models:
        # Parent Run for the Model Type
        with mlflow.start_run(run_name=model_name):
            
            # 1. Hyperparameter Tuning with bayesian optimization
            logging.info(f"MODEL_TRAIN: Tuning {model_name}")
            study = optuna.create_study(direction="maximize")
            study.optimize(_tune_model(model_name, X_train, y_train), n_trials=n_trials)
            
            try:
                fig_hist = optuna.visualization.plot_optimization_history(study)
                fig_slice = optuna.visualization.plot_slice(study)
                
                # Log to MLflow as interactive HTML artifacts
                mlflow.log_figure(fig_hist, f"plots/{model_name}_optimization_history.html")
                mlflow.log_figure(fig_slice, f"plots/{model_name}_slice_plot.html")
                
                logging.info(f"MODEL_TRAIN: log historic trials plot for {model_name} study")
            except Exception as e:
                logging.warning(f"MODEL_TRAIN: failed to generate plots: {e}")
                
            # 2. Collect and Train model with Best Params
            best_params = study.best_params.copy()
            match model_name:
                case "randomforest": final_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
                case "xgboost": final_model = XGBClassifier(**best_params, random_state=RANDOM_STATE)
                case "hgboost": final_model = HistGradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
                case "catboost": final_model = CatBoostClassifier(**best_params, random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
                case "lightgbm": final_model = LGBMClassifier(**best_params, random_state=RANDOM_STATE)
                case _: raise ValueError(f"Unsupported model: {model_name}")
            final_model.fit(X_train, y_train)
            
            # 3. Log model metrics
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "recall": recall_score(y_test, y_pred),
                "avg_precision": average_precision_score(y_test, y_prob),
                "auc_roc": roc_auc_score(y_test, y_prob),
                "accuracy": accuracy_score(y_test, y_pred),
            }
            mlflow.log_metrics(metrics)
            _evaluate_model(final_model, X_test, y_test, y_pred)
            
            # 4. Log best model and params
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(final_model, artifact_path="model")