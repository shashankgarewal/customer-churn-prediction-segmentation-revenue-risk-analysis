import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import mlflow

from src.utils.common import get_project_root
from src.utils.logger import logging
from src.model.evaluate_churn import track_model, _assign_segment, plot_segment_distributions

from src.utils.config import RANDOM_STATE



# -------------------------------------- hyperparmeter setup ------------------------------------- #
def _tune_model(model_name, X, y):
    """hyperparameter setup with optuna and cross-validation"""

    # Multi-level stratification: combine segment and label once per study
    strat_key = X['Lifetime_Value'].apply(_assign_segment).astype(str) + "_" + y.astype(str)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X, strat_key))

    def objective(trial):
        match model_name:
            case "randomforest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "class_weight": None,
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1
                }
                model = RandomForestClassifier(**params)
            case "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 550, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.02),
                    "max_depth": trial.suggest_int("max_depth", 6, 8),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "random_state": RANDOM_STATE
                }
                model = XGBClassifier(**params)

            case "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 600, 900),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "depth": trial.suggest_int("depth", 6, 9),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 4, 9),
                    "random_seed": RANDOM_STATE,
                    "verbose": False,
                    "allow_writing_files": True
                }
                model = CatBoostClassifier(**params)

            case "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 550, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.04),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 70),
                    "max_depth": trial.suggest_int("max_depth", 10, 20),
                    "random_state": RANDOM_STATE
                }
                model = LGBMClassifier(**params)

            case "hgboost":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06),
                    "max_iter": trial.suggest_int("max_iter", 400, 500),
                    "max_depth": trial.suggest_int("max_depth", 8, 10),
                    "random_state": RANDOM_STATE
                }
                model = HistGradientBoostingClassifier(**params)
            case _:
                raise ValueError(f"Unsupported model for tuning: {model_name}")

        fit_params = {}
        if model_name == "catboost":
            fit_params["cat_features"] = X.select_dtypes(include=["object", "category"]).columns.tolist()
            
        scores = cross_val_score(
            model, X, y, 
            cv=cv_splits,  
            scoring="neg_log_loss", 
            n_jobs=-1, 
            params=fit_params if fit_params else None,
        )
        
        return scores.mean()
    
    return objective


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
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root
        )
    elif exp.lifecycle_stage == "deleted":
        logging.info("MODEL_TRAIN: Experiment '%s' exists but is deleted. Restoring it.", experiment_name)
        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)

    mlflow.set_experiment(experiment_name)
    
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]
    
    if not models:
        models = ["randomforest", "xgboost", "hgboost", "catboost", "lightgbm"]
    
    for model_name in models:
        # Psarent Run for the Model Type
        with mlflow.start_run(run_name=model_name):
            
            # 1. Hyperparameter Tuning with bayesian optimization
            logging.info(f"MODEL_TRAIN: Tuning {model_name} with {n_trials} experiments")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                _tune_model(model_name, X_train, y_train), 
                n_trials=n_trials, 
                show_progress_bar=True
            )
            mlflow.log_param("n_trials", len(study.trials))
            
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
            cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
            match model_name:
                case "randomforest": final_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
                case "xgboost": final_model = XGBClassifier(**best_params, random_state=RANDOM_STATE)
                case "hgboost": final_model = HistGradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
                case "catboost": final_model = CatBoostClassifier(**best_params, 
                                                                  cat_features=cat_cols, random_seed=RANDOM_STATE, 
                                                                  verbose=False, allow_writing_files=False)
                case "lightgbm": final_model = LGBMClassifier(**best_params, random_state=RANDOM_STATE)
                case _: raise ValueError(f"Unsupported model: {model_name}")
            final_model.fit(X_train, y_train)
            
            # 3. Log model metrics including segment-specific results
            metrics = track_model(final_model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # 4. Generate and log probability distribution plots per segment
            dist_figs = plot_segment_distributions(final_model, X_test, y_test)
            for seg_name, fig in dist_figs.items():
                mlflow.log_figure(fig, f"plots/{model_name}_{seg_name.lower()}_distribution.png")
                plt.close(fig)
            
            # 5. Log best model and params
            mlflow.log_params(best_params)
            logging.info(f"MODEL_TRAIN: log metric, probability distribution, and parameters")
            match model_name:
                case "randomforest" | "hgboost":
                    mlflow.sklearn.log_model(
                        final_model,
                        name=model_name
                    )

                case "xgboost":
                    mlflow.xgboost.log_model(
                        final_model,
                        name=model_name
                    )

                case "catboost":
                    mlflow.catboost.log_model(
                        final_model,
                        name=model_name
                    )

                case "lightgbm":
                    mlflow.lightgbm.log_model(
                        final_model,
                        name=model_name
                    )
            mlflow.set_tag("model_type", model_name)
                