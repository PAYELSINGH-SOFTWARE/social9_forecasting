from pathlib import Path


# ==============================
# PROJECT DIRECTORIES
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# ==============================
# DATA FILES
# ==============================



MODEL_FILE = MODEL_DIR / "social9_xgboost.json"

METADATA_FILE = MODEL_DIR / "metadata.json"


# ==============================
# FORECAST SETTINGS
# ==============================

DEFAULT_FORECAST_DAYS = 30

MAX_FORECAST_DAYS = 90

MIN_FORECAST_DAYS = 1


# ==============================
# MODEL FEATURES
# ==============================

FEATURES = [
    "likes",
    "comments",
    "shares",
    "reach",
    "impressions",
    "followers",
    "posts_count",
    "engagement_rate",
    "day_of_week",
    "day_of_month",
    "month",
    "lag_1",
    "lag_7",
    "rolling_7",
]


# ==============================
# TARGET
# ==============================

TARGET = "engagement"


# ==============================
# REQUIRED DATA COLUMNS
# ==============================

REQUIRED_COLUMNS = [
    "instagram_id",
    "date",
    "likes",
    "comments",
    "shares",
    "reach",
    "impressions",
    "followers",
    "posts_count",
    "engagement",
]