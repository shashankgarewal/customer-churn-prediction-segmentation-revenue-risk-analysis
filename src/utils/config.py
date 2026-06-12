import yaml
from pathlib import Path
from src.utils.common import get_project_root

PROJECT_ROOT = get_project_root()
CONFIG_PATH = PROJECT_ROOT / "config" / "production.yaml"

def _load_config():
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

CONFIG = _load_config()

# Helper/exposed variables to keep imports simple and clean
# raw_data
RAW_DATA_PATH = CONFIG.get("raw_data", {}).get("path", "data/raw/ecommerce_customer_churn_dataset.csv")
RAW_DATA_COLUMNS = CONFIG.get("raw_data", {}).get("columns", [])

# pipeline
TARGET = CONFIG.get("pipeline", {}).get("target", "Churned")
DEFAULT_MODELS = CONFIG.get("pipeline", {}).get("default_models", ["randomforest", "xgboost", "hgboost", "catboost", "lightgbm"])

# model_registry
MODEL_NAME = CONFIG.get("model_registry", {}).get("model_name", "churn_model")
ALIAS = CONFIG.get("model_registry", {}).get("alias", "best")

# features
LOW_CARDINAL_COLS = CONFIG.get("features", {}).get("low_cardinal_cols", ["Gender", "Signup_Quarter", "Country"])
MID_CARDINAL_COLS = CONFIG.get("features", {}).get("mid_cardinal_cols", ["City"])
NEW_FEATURES = CONFIG.get("features", {}).get("new_features", ["Engagement_Score", "Purchase_Frequency", "LTV_Per_Purchase"])

# data_preprocess
TRAIN_SIZE = CONFIG.get("data_preprocess", {}).get("train_size", 0.8)
RANDOM_STATE = CONFIG.get("data_preprocess", {}).get("random_state", 42)

# retention
EXCLUDE_FROM_DRIVERS = CONFIG.get("retention", {}).get("exclude_from_drivers", ["Lifetime_Value"])
RETENTION_RATES = CONFIG.get("retention", {}).get("retention_rates", {
    "VIP": 0.85, "IMP": 0.80, "High": 0.75, "Medium": 0.65, "Low": 0.50, "No_VALUE": 0.30
})
# Make sure threshold keys are parsed correctly if loaded as list/dict
THRESHOLDS = CONFIG.get("retention", {}).get("thresholds", {
    "VIP": [4400, 0.18], "IMP": [2600, 0.2], "High": [1800, 0.4], "Medium": [800, 0.4], "Low": [0, 0.55]
})
PERSONA_RULES = CONFIG.get("retention", {}).get("persona_rules", {})
ACTION_PLAYBOOK = CONFIG.get("retention", {}).get("action_playbook", {})

# app
API_BASE = CONFIG.get("app", {}).get("api_base", "http://localhost:8000")
SEGMENT_ORDER = list(THRESHOLDS.keys())
BRAND_COLORS = CONFIG.get("app", {}).get("brand_colors", ["#c0785a", "#78a898", "#b0a898", "#6b6259", "#c4baae"])
