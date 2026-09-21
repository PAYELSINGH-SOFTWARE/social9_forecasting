from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "social9_engagement.csv"

MODEL_FILE = MODEL_DIR / "social9_xgboost.json"

FEATURES = [
    "day_of_week",
    "month",
    "lag_1_ratio",
    "lag_7_ratio",
    "rolling_std_ratio",
    "recent_trend",
    "lag_posts_1",
    "posting_rate_7",
    "posts_rolling_7",
    "days_since_post",
]

TARGET = "engagement_ratio"
