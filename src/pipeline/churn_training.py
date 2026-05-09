from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data.ingest import load_data
from src.data.preprocess import prepare_dataset
from src.data.validate import validate_data
from src.features.create import interaction_features
from src.features.transform import (
    encode_low_cardinality_features,
    encode_mid_cardinality_features,
    impute_missing_features,
)
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


def _create_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = interaction_features(train_df)
    test_features = interaction_features(test_df)
    _log_step_shape("feature_creation", train_features, test_features)
    return train_features, test_features


def _transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_processed = impute_missing_features(train_df, fit=True)
    test_processed = impute_missing_features(test_df, fit=False)

    train_processed = encode_low_cardinality_features(train_processed, fit=True)
    test_processed = encode_low_cardinality_features(test_processed, fit=False)

    train_processed = encode_mid_cardinality_features(train_processed, fit=True)
    test_processed = encode_mid_cardinality_features(test_processed, fit=False)

    feature_columns = [column for column in train_processed.columns if column != TARGET]
    ordered_columns = feature_columns + [TARGET]

    train_processed = train_processed.reindex(columns=ordered_columns)
    test_processed = test_processed.reindex(columns=ordered_columns, fill_value=0)

    _log_step_shape("feature_transformation", train_processed, test_processed)
    return train_processed, test_processed


def run_pipeline(
    raw_data_path: str | Path = DEFAULT_RAW_DATA_PATH,
    models: Sequence[str] | None = None,
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

    train_features, test_features = _create_features(train_df, test_df)
    _save_split(train_features, test_features, artifacts.feature_dir)

    train_processed, test_processed = _transform_features(train_features, test_features)
    _save_split(train_processed, test_processed, artifacts.transformed_dir)

    logging.info("PIPELINE_STEP: model_training started | models=%s", selected_models)
    build_model(train_df=train_processed, test_df=test_processed, target=TARGET, models=selected_models)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(raw_data_path=args.data_path, models=args.models)
