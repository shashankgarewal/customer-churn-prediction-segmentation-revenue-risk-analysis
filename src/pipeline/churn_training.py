from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data.ingest import load_data
from src.data.preprocess import prepare_dataset
from src.data.validate import validate_data
from src.features.processor import process_features
from src.model.train_churn import build_model
from src.utils.logger import logging


TARGET = "Churned"
DEFAULT_RAW_DATA_PATH = Path("data/raw/ecommerce_customer_churn_dataset.csv")
DEFAULT_MODELS = ["randomforest", "xgboost", "hgboost", "catboost", "lightgbm"]


@dataclass(frozen=True)
class PipelineArtifacts:
    raw_path: Path
    base_dir: Path = Path("data/interim/base")
    feature_dir: Path = Path("data/interim/new-features")
    imputed_dir: Path = Path("data/interim/imputed")
    encoded_dir: Path = Path("data/interim/encoded")
    processed_dir: Path = Path("data/processed")
    transformed_dir: Path = Path("data/interim/transformed")


def _save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    logging.info("PIPELINE_SAVE: saved train/test split to %s", output_dir)


def _log_step_shape(step_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    logging.info(
        "PIPELINE_STEP: %s complete | train_shape=%s | test_shape=%s",
        step_name,
        train_df.shape,
        test_df.shape,
    )


def _discard_item(items: list[str], item: str) -> list[str]:
    """Remove an item from the list and return the remaining items."""
    return [i for i in items if i != item]

def run_pipeline(
    raw_data_path: str | Path = DEFAULT_RAW_DATA_PATH,
    models: Sequence[str] | None = None,
    trials: int = 15
) -> None:
    """Run the full churn training workflow from raw data to model experiments."""

    artifacts = PipelineArtifacts(raw_path=Path(raw_data_path))
    selected_models = list(models) if models else DEFAULT_MODELS

    logging.info("PIPELINE_START: loading raw dataset from %s", artifacts.raw_path)
    raw_df = load_data(str(artifacts.raw_path))

    validated_df = validate_data(raw_df)
    logging.info("PIPELINE_STEP: validation complete | dataset_shape=%s", validated_df.shape)

    train_df, test_df = prepare_dataset(validated_df)
    _save_split(train_df, test_df, artifacts.base_dir)
    _log_step_shape("dataset_split", train_df, test_df)

    # Unified processing for CatBoost (skips encoding)
    train_preencoded_cat = process_features(train_df, fit=True, is_catboost=True)
    test_preencoded_cat = process_features(test_df, fit=False, is_catboost=True)
    _log_step_shape("feature_processing_catboost", train_preencoded_cat, test_preencoded_cat)
    _save_split(train_preencoded_cat, test_preencoded_cat, artifacts.transformed_dir)

    # Unified processing for all other models (includes encoding)
    train_processed = process_features(train_df, fit=True, is_catboost=False)
    test_processed = process_features(test_df, fit=False, is_catboost=False)
    _log_step_shape("feature_processing_standard", train_processed, test_processed)
    _save_split(train_processed, test_processed, artifacts.processed_dir)

    logging.info("PIPELINE_STEP: model_training started | models=%s", selected_models)
    
    if "catboost" in selected_models:
        build_model(
            train_df=train_preencoded_cat,
            test_df=test_preencoded_cat,
            target=TARGET,
            models=["catboost"],
            n_trials=trials,
        )

    rest_models = _discard_item(selected_models, "catboost")

    if rest_models:
        build_model(
            train_df=train_processed,
            test_df=test_processed,
            target=TARGET,
            models=rest_models,
            n_trials=trials,
        )
    logging.info("PIPELINE_COMPLETE: churn training workflow finished successfully.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the churn training pipeline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the raw churn dataset CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model names to train.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="No of trials per model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(raw_data_path=args.data_path, models=args.models, trials=args.trials)
