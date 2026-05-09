import pandas as pd
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
)
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.utils.common import get_project_root
from src.utils.logger import logging

RANDOM_STATE = 42 

# ------------------------------------ mlflow experiment setup ----------------------------------- #
def _setup_mlflow(
    experiment_name: str = "churn-training-pipeline",
):
    project_root = get_project_root()
    artifact_root = f"file://{project_root}/mlflow/experiment_logs"
    tracking_db = f"sqlite:///{project_root}/mlflow/tracking.db"

    mlflow.set_tracking_uri(tracking_db)

    # set log_model and log_artifact location at create experiment
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root
        )

    mlflow.set_experiment(experiment_name)


# -------------------------------------- hyperparmeter setup ------------------------------------- #
def _tune_model(X, y):
    """hyperparameter setup using optuna and cross-validation"""
    def xgboost_objective(trial):
        xgboost_params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()
    params = {"xgboost": {}
        
    }
    
    study = optuna.create_study(direction="maximize")
    study.optimize(xgboost_objective, n_trials=20)
    return study.best_params

# ------------------------------- evaluate model and log in mlflow ------------------------------- #
def _evaluate_model():
    """evaluate and log model metrics in mlflow"""
    return

def build_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "Churned",
):
    """build model"""
    _setup_mlflow()
    
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]
    
    results: list[dict[str, float | str]] = []
    
    best_model  = None
    best_recall = -1.0
    
    return
