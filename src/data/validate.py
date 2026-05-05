import pandas as pd
import numpy as np
import great_expectations as gx
import great_expectations.expectations as gxe

def validate_data(df: pd.DataFrame):
    """This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    Args:
        df (pd.DataFrame): raw dataframe
    """
    
    # -------------------- govern by data-validation notebook -------------------- #
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
    
    # ----------- unobserved invalid cases not observed in known data. ----------- #
    
    ## connect pandas df in gx
    context = gx.get_context()
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="data asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    
    