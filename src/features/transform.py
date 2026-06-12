import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.utils.common import get_project_root
from src.utils import logger


PROJECT_ROOT = get_project_root()
ARTIFACT_PATH = PROJECT_ROOT / "artifacts"
ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)


from src.utils.config import LOW_CARDINAL_COLS, MID_CARDINAL_COLS, TARGET



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for frequency encoding."""

    def __init__(self):
        self.maps_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            # Store normalized frequencies
            self.maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.maps_.items():
            if col in X.columns:
                unseen = X[~X[col].isin(mapping.keys())][col].unique()
                if len(unseen) > 0:
                    logger.logging.warning("FrequencyEncoder: Unseen categories in %s column: %s", col, unseen)
                # Map raw values to frequencies; fill unseen values with 0
                X[col] = X[col].map(mapping).fillna(0)
        return X


def impute_missing_features(
    df: pd.DataFrame, fit: bool = False
) -> pd.DataFrame:
    """Median-impute numeric columns using persisted statistics."""
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    file_path = ARTIFACT_PATH / "median_imputer.joblib"

    try:
        if fit:
            logger.logging.info(f"MedianImputer: Fitting on {len(numeric_cols)} numeric columns: {numeric_cols}")
            imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
            imputer.fit(df[numeric_cols])
            joblib.dump(imputer, file_path)
            logger.logging.info(f"MedianImputer: Saved at [{file_path.relative_to(PROJECT_ROOT)}]")

        else:
            logger.logging.info("MedianImputer: Loading from %s", file_path)
            imputer = joblib.load(file_path)
            
    except Exception as e:
        logger.logging.error("MedianImputer: Failed to process. Error: %s", str(e))
        raise e

    imputed_data = imputer.transform(df[numeric_cols])
    df = df.combine_first(imputed_data)
    return df


def encode_low_cardinality_features(
    df: pd.DataFrame, fit: bool = False, columns: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode low-cardinality categorical columns."""
    
    if columns is None:
        columns = LOW_CARDINAL_COLS
    
    df = df.copy()
    
    available_cols = [col for col in columns if col in df.columns]
    file_path = ARTIFACT_PATH / "onehot_encoder.joblib"

    try:
        if fit:
            logger.logging.info(f"OneHotEncoder: Fitting on {len(available_cols)} columns: {available_cols}")
            
            encoder = OneHotEncoder(
                sparse_output=False, drop="first", dtype=np.uint8, handle_unknown="error"
            ).set_output(transform="pandas")
            encoder.fit(df[available_cols])
            joblib.dump(encoder, file_path)
            
            logger.logging.info(f"OneHotEncoder: Saved at [{file_path.relative_to(PROJECT_ROOT)}]")
        else:
            logger.logging.info(f"OneHotEncoder: Loading from [{file_path.relative_to(PROJECT_ROOT)}]")
            
            encoder = joblib.load(file_path)
    except Exception as e:
        logger.logging.error("OneHotEncoder: Failed to process. Error: %s", str(e))
        raise e

    encoded_df = encoder.transform(df[available_cols])
    df = pd.concat([df, encoded_df], axis=1).drop(columns=available_cols)
    return df


def encode_mid_cardinality_features(
    df: pd.DataFrame, fit: bool = False, columns: list[str] | None = None
) -> pd.DataFrame:
    """Frequency encode mid-cardinality categorical columns using custom transformer."""
    
    if columns is None:
        columns = MID_CARDINAL_COLS

    df = df.copy()
    available_cols = [col for col in columns if col in df.columns]
    
    if not available_cols:
        return df

    file_path = ARTIFACT_PATH / "frequency_encoder.joblib"

    try:
        if fit:
            logger.logging.info(f"FrequencyEncoder: Fitting on {len(available_cols)} columns: {available_cols}")
            encoder = FrequencyEncoder()
            encoder.fit(df[available_cols])
            joblib.dump(encoder, file_path)
            logger.logging.info(f"FrequencyEncoder: Saved at [{file_path.relative_to(PROJECT_ROOT)}]")
            
        else:
            logger.logging.info(f"FrequencyEncoder: Loading from [{file_path.relative_to(PROJECT_ROOT)}]")
            encoder = joblib.load(file_path)
            
    except Exception as e:
        logger.logging.error("FrequencyEncoder: Failed with error %s", str(e))
        raise e

    df[available_cols] = encoder.transform(df[available_cols])
    return df