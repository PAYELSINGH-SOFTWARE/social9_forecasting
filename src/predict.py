from datetime import timedelta

import pandas as pd
import xgboost as xgb

from .config import (
    FEATURES,
    MODEL_FILE,
)

from .data_loader import load_data
from .preprocessing import clean_data


HISTORY_COLUMNS = [
    "date",
    "likes",
    "comments",
    "shares",
    "reach",
    "impressions",
    "followers",
    "posts_count",
]


def _history_frame(history: list[dict] | None) -> pd.DataFrame:
    if history is None:
        frame = load_data()
        if "account_id" in frame.columns:
            account_ids = sorted(frame["account_id"].unique())
            frame = frame[frame["account_id"] == account_ids[len(account_ids) // 2]]
        return frame.reset_index(drop=True)
    if len(history) < 8:
        raise ValueError("At least 8 days of history are required")
    frame = pd.DataFrame(history, columns=HISTORY_COLUMNS)
    frame["account_id"] = "live_account"
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("History contains an invalid date")
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_trained_model():

    model = xgb.XGBRegressor()

    model.load_model(
        str(MODEL_FILE)
    )

    return model


def forecast(
    forecast_days=7,
    history: list[dict] | None = None,
):
    df = _history_frame(history)

    df = clean_data(df)

    df = df.sort_values("date").reset_index(drop=True)

    model = load_trained_model()

    results = []

    working_df = df.copy()

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

        recent = engagement.tail(7)
        rolling_7 = recent.mean()

        if rolling_7 <= 0:
            prediction = 0.0
        else:
            row = {
                "day_of_week": next_date.dayofweek,
                "month": next_date.month,
                "lag_1_ratio": lag_1 / rolling_7,
                "lag_7_ratio": lag_7 / rolling_7,
                "rolling_std_ratio": recent.std(ddof=0) / rolling_7,
                "recent_trend": (lag_1 + 1) / (lag_7 + 1),
            }

            X = pd.DataFrame([row], columns=FEATURES)
            predicted_ratio = float(model.predict(X)[0])
            predicted_ratio = min(2.5, max(0.0, predicted_ratio))
            prediction = predicted_ratio * rolling_7

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
