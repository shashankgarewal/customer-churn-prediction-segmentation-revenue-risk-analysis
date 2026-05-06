import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe

from src.utils import logger, exception

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "production.yaml"

def _get_df_columns(df: pd.DataFrame) -> list[str]:
    """Load config, or save config with columns if not present. Return columns in df as list."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as prod:
            config = yaml.safe_load(prod) or {}
    except FileNotFoundError:
        config = {}
        logger.logging.warning(
            "CONFIG_MISSING: %s not found. Current dataframe columns will be used.",
            CONFIG_PATH,
        )

    columns = config.get("raw_data", {}).get("columns", [])

    if not columns:
        columns = df.columns.to_list()
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as prod:
            yaml.safe_dump(
                {"raw_data": {"columns": columns}},
                prod,
                default_flow_style=False,
                sort_keys=False,
            )
            
    return columns
    
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    Args:
        df (pd.DataFrame): raw dataframe
    """

    # ----------------------------------------- inital setup ----------------------------------------- #
    ## connect pandas df in gx
    batch = gx.get_context().data_sources.pandas_default.read_dataframe(df)
    
    columns = _get_df_columns()
    
    # -------------------------------- column presence and order check ------------------------------- #
    for col in columns: 
        order_result = gxe.ExpectColumnToExist(column=col)
        if not order_result.success:
            logger.logging.warning(
                "SCHEMA_MISMATCH: missing column '%s' created with null values.", 
                col
            )
            df[col] = np.nan
            
    df = df.reindex(columns=columns).copy()

    # ------------------------------ govern by data-validation notebook ------------------------------ #
    ## dropped early, not worth intervene even if churn. 
    ## ltv case (no purchase and no value), and age case (tiny set and no signal)
    invalid_mask = (
        df['Age'].gt(100).fillna(False) |
        df['Lifetime_Value'].lt(0).fillna(False)
    )
    df = df[~invalid_mask]
 
    ## flag invalid - reasonable size to keep with flag for pattern
    invalid_flag_rules = {
        'Flag_Purchases_Invalid': df['Total_Purchases'].lt(0),
        'Flag_Discount_Rate_Invalid': df['Discount_Usage_Rate'].gt(100),
        'Flag_Membership_Age_Invalid': df['Membership_Years'].gt(df['Age']),
    }

    for flag_col, condition in invalid_flag_rules.items():
        df[flag_col] = condition.fillna(False).astype('uint8')
    
    # ------------------------------ range check not observed in dataset ----------------------------- #
    gx_expectations = [
        gxe.ExpectColumnValuesToBeBetween(column="Membership_Years", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="Days_Since_Last_Purchase", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="Wishlist_Items", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="Customer_Service_Calls", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="Product_Reviews_Written", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="Cart_Abandonment_Rate", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeBetween(column="Returns_Rate", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeBetween(column="Email_Open_Rate", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeBetween(column="Social_Media_Engagement_Score", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeInSet(column="Signup_Quarter", value_set=["Q1", "Q2", "Q3", "Q4"]),
        gxe.ExpectColumnValuesToBeInSet(column="Churned", value_set=[0, 1]),
    ]

    for gx_expectation in gx_expectations:
        result = batch.validate(gx_expectation)
        if not result.success:
            logger.logging.warning(
                "GX_VALIDATION_FAILED: %s failed for '%s' (unexpected_count=%s, sample=%s)",
                result.expectation_config.type,
                result.expectation_config.kwargs.get("column", "unknown"),
                result.result.get("unexpected_count", 0),
                result.result.get("partial_unexpected_list", []),
            )

    logger.logging.info("DATA_VALIDATION_COMPLETED: validation finished with shape %s.", df.shape)
    return df

    
