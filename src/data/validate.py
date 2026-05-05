import pandas as pd
import numpy as np
import yaml
import os
import great_expectations as gx
import great_expectations.expectations as gxe

from src.utils import logger, exception


def validate_data(df: pd.DataFrame):
    """This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    Args:
        df (pd.DataFrame): raw dataframe
    """

    # ----------------------------------------- inital setup ----------------------------------------- #
    ## connect pandas df in gx
    context = gx.get_context()
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="data asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    ## load config
    config_path = '/config/production.yaml'
    try:
        with open(config_path, 'r') as prod:
            config = yaml.safe_load(prod)
    except FileNotFoundError:
        config = {}
    except Exception as e:
        raise e
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with (config_path, 'w') as prod:
        config = {'data': {'columns': df.columns.to_list()}}
        yaml.safe_dump(config, prod, default_flow_style=False)

    columns = config.get('data', {}).get('columns', [])
        
    
    # -------------------------------- column presence and order check ------------------------------- #
    for col in columns: 
        order_result = gxe.ExpectColumnToExist(column=col)
        if not order_result.success:
            logger.warning("SCHEMA_MISMATCH: Column order does not match the production.yaml definition.")
            df[col] = 0
            
    df = df.reindex(columns=columns)
    
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
    
    # ----------- unobserved invalid cases not observed in known data. ----------- #
    

    
    