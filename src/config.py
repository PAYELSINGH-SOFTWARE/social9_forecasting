from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "social9_engagement.csv"

MODEL_FILE = MODEL_DIR / "social9_xgboost.json"

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

TARGET = "engagement"