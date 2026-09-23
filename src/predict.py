from datetime import timedelta

import pandas as pd
import xgboost as xgb

from .config import FEATURES, MODEL_FILE
from .api_data_loader import load_data_from_api
from .preprocessing import clean_data
from .features import create_features


# =========================
# LOAD TRAINED MODEL
# =========================

def load_trained_model():
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_FILE))
    return model


# =========================
# FILTER INSTAGRAM DATA
# =========================

def filter_instagram_data(df, instagram_id=None):

    if instagram_id is None:
        return df

    if "instagram_id" not in df.columns:
        raise ValueError(
            "Instagram ID column not found in dataset."
        )

    filtered_df = df[
        df["instagram_id"].astype(str).str.strip()
        == str(instagram_id).strip()
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No data found for Instagram ID: {instagram_id}"
        )

    return filtered_df


# =========================
# FORECASTING FUNCTION
# =========================

def forecast(
    historical_data: list[dict],
    forecast_days: int = 30,
    instagram_id: str | None = None
):

    if forecast_days < 1 or forecast_days > 90:
        raise ValueError(
            "forecast_days must be between 1 and 90."
        )

    if not historical_data:
        raise ValueError(
            "Historical data cannot be empty."
        )

    # Load data from API request
    df = load_data_from_api(historical_data)

    # Filter Instagram account
    df = filter_instagram_data(df, instagram_id)

    # Clean data
    df = clean_data(df)

    # Create features
    df = create_features(df)

    # Remove missing feature values
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid data available after preprocessing. "
            "At least 8 historical records are recommended."
        )

    # Load trained model
    model = load_trained_model()

    results = []
    working_df = df.copy()

    # Generate forecast
    for _ in range(forecast_days):

        next_date = (
            working_df["date"].iloc[-1]
            + timedelta(days=1)
        )

        last_row = working_df.iloc[-1]
        engagement = working_df["engagement"]

        lag_1 = engagement.iloc[-1]

        lag_7 = (
            engagement.iloc[-7]
            if len(engagement) >= 7
            else engagement.mean()
        )

        rolling_7 = engagement.tail(7).mean()

        row = {
            "likes": last_row["likes"],
            "comments": last_row["comments"],
            "shares": last_row["shares"],
            "reach": last_row["reach"],
            "impressions": last_row["impressions"],
            "followers": last_row["followers"],
            "posts_count": last_row["posts_count"],
            "engagement_rate": last_row["engagement_rate"],
            "day_of_week": next_date.dayofweek,
            "day_of_month": next_date.day,
            "month": next_date.month,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_7": rolling_7,
        }

        X = pd.DataFrame(
            [row],
            columns=FEATURES
        )

        prediction = float(model.predict(X)[0])
        prediction = max(0.0, prediction)

        account_id = (
            str(instagram_id)
            if instagram_id is not None
            else str(last_row["instagram_id"])
        )

        results.append({
            "instagram_id": account_id,
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_engagement": round(
                prediction,
                2
            ),
        })

        # Update working data for the next prediction
        new_row = last_row.copy()

        new_row["date"] = next_date
        new_row["engagement"] = prediction

        working_df = pd.concat(
            [
                working_df,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

    return results


# =========================
# TEST FORECASTING
# =========================

if __name__ == "__main__":

    sample_data = [
        {
            "instagram_id": "account_001",
            "date": "2026-09-01",
            "likes": 100,
            "comments": 20,
            "shares": 10,
            "reach": 1000,
            "impressions": 1500,
            "followers": 5000,
            "posts_count": 2
        }
    ]

    predictions = forecast(
        historical_data=sample_data,
        forecast_days=30,
        instagram_id="account_001"
    )

    for prediction in predictions:
        print(
            prediction["date"],
            "→",
            prediction["predicted_engagement"]
        )