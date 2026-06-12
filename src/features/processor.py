import pandas as pd
from src.features.create import interaction_features
from src.features.transform import (
    encode_low_cardinality_features,
    encode_mid_cardinality_features,
    impute_missing_features,
)

from src.utils.config import TARGET


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates interaction features."""
    return interaction_features(df.copy())

def impute_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Handles missing values using stored or fitted statistics."""
    return impute_missing_features(df.copy(), fit=fit)

def encode_categorical(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Encodes categorical variables and ensures consistent column ordering."""
    df = df.copy()
    df = encode_low_cardinality_features(df, fit=fit)
    df = encode_mid_cardinality_features(df, fit=fit)

    # Reorder columns to ensure TARGET is last if present
    if TARGET in df.columns:
        feature_columns = [col for col in df.columns if col != TARGET]
        df = df.reindex(columns=feature_columns + [TARGET])
    
    return df

def process_features(df: pd.DataFrame, fit: bool = False, is_catboost: bool = False) -> pd.DataFrame:
    """
    Complete data transformation pipeline for inference or training.
    
    Args:
        df: Input dataframe.
        fit: Whether to fit transformers (True for training, False for inference).
        is_catboost: If True, skips categorical encoding as CatBoost handles it natively.
    """
    df = create_features(df)
    df = impute_features(df, fit=fit)
    if not is_catboost:
        df = encode_categorical(df, fit=fit)
    return df