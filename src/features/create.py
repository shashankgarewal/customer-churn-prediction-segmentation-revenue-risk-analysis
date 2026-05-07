import pandas as pd
from src.utils import logger


NEW_FEATURES = [
    "Engagement_Score",
    "Purchase_Frequency",
    "LTV_Per_Purchase",
]


def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific interaction features."""
    
    df = df.copy()

    df["Engagement_Score"] = df["Pages_Per_Session"] * df["Session_Duration_Avg"]
    
    df["Purchase_Frequency"] = (
        df["Total_Purchases"] /
        (df["Days_Since_Last_Purchase"].clip(lower=0) + 1)
    )
    
    df["LTV_Per_Purchase"] = (
        df["Lifetime_Value"] /
        (df["Total_Purchases"].clip(lower=0) + 1)
    )
    logger.logging.info("INTERACTION_FEATURES: create %s new features: %s", len(NEW_FEATURES), NEW_FEATURES)

    return df
