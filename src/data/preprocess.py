import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import logger
from src.utils.common import get_project_root

PROJECT_ROOT = get_project_root()
PREPROCESS_DATA_PATH = PROJECT_ROOT / "data" / "preprocess" 
TRAIN_SIZE = 0.8
TARGET = 'Churned'

def prepare_dataset(df: pd.DataFrame):
    """Prepare the dataset for feature engineering"""
    PREPROCESS_DATA_PATH.mkdir(parents=True, exist_ok=True)

     # ------------------------------- preprocess flags (distributional) ------------------------------ #
    # Missing Flag: Captures rows with any nulls (or specific important ones)
    df['Flag_Missing'] = df.isnull().any(axis=1).astype('uint8')
    
    # Extreme Flag: Statistical outlier detection
    df['Flag_AOV_Extreme'] = df["Average_Order_Value"].gt(400).fillna(False).astype('uint8')
    
    logger.logging.info("FLAG: create [Flag_Missing, Flag_AOV_Extreme] flag features")
    
    # ---------------------------------- split data to avoid leakage --------------------------------- #
    train_df, test_df = train_test_split(
        df, 
        train_size=TRAIN_SIZE, 
        stratify=df[TARGET],
        random_state=42
        )
    
    # ---------------------------------- save train and test dataset --------------------------------- #
    train_path = PREPROCESS_DATA_PATH / "train.parquet"
    test_path = PREPROCESS_DATA_PATH / "test.parquet"
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    # ----------------------------------- consolidated log message ----------------------------------- #
    log_metrics = ["Dataset Split Summary:"]
    log_metrics.append(f"  Total Records: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    
    flag_cols = [col for col in df.columns if col.startswith('Flag_')]
    for flag_col in flag_cols:
        train_sum = train_df[flag_col].sum()
        test_sum = test_df[flag_col].sum()
        log_metrics.append(f"  {flag_col} -> train: {train_sum}, test: {test_sum}")
    
    log_metrics.append("\nPREPARE_TRAIN_TEST: Saved dataset ready for feature engineering:")
    log_metrics.append(f"  Train: [{train_path.relative_to(PROJECT_ROOT)}]")
    log_metrics.append(f"  Test:  [{test_path.relative_to(PROJECT_ROOT)}]")
    
    logger.logging.info("\n".join(log_metrics))
    
    return train_df, test_df