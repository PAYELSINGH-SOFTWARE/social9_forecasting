from datetime import timedelta

import pandas as pd
import xgboost as xgb

from .config import (
    FEATURES,
    MODEL_FILE,
)

from .data_loader import load_data
from .preprocessing import clean_data
from .features import create_features


def load_trained_model():

    model = xgb.XGBRegressor()

    model.load_model(
        str(MODEL_FILE)
    )

    return model


def forecast(
    forecast_days=7
):

    df = load_data()

    df = clean_data(df)

    df = create_features(df)

    df = df.dropna().reset_index(
        drop=True
    )

    model = load_trained_model()

    results = []

    working_df = df.copy()

    for _ in range(forecast_days):

        next_date = (
            working_df["date"].iloc[-1]
            + timedelta(days=1)
        )

        last_row = working_df.iloc[-1]

        engagement = (
            working_df["engagement"]
        )

        lag_1 = engagement.iloc[-1]

        lag_7 = (
            engagement.iloc[-7]
            if len(engagement) >= 7
            else engagement.mean()
        )

        rolling_7 = (
            engagement.tail(7).mean()
        )

        row = {
            "likes": last_row["likes"],
            "comments": last_row["comments"],
            "shares": last_row["shares"],
            "reach": last_row["reach"],
            "impressions": last_row["impressions"],
            "followers": last_row["followers"],
            "posts_count": last_row["posts_count"],
            "engagement_rate": last_row[
                "engagement_rate"
            ],
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

        prediction = model.predict(X)[0]

        prediction = max(
            0,
            float(prediction)
        )

        results.append({
            "date": next_date.strftime(
                "%Y-%m-%d"
            ),
            "predicted_engagement": round(
                prediction,
                2
            )
        })

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


if __name__ == "__main__":

    predictions = forecast(7)

    for prediction in predictions:

        print(
            prediction["date"],
            "→",
            prediction[
                "predicted_engagement"
            ]
        )