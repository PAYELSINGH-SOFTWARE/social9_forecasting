def create_features(df):

    df = df.copy()

    if "account_id" not in df.columns:
        df["account_id"] = "account_1"

    df = df.sort_values(
        ["account_id", "date"]
    ).reset_index(drop=True)

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["month"] = (
        df["date"].dt.month
    )

    engagement = df.groupby("account_id", sort=False)["engagement"]
    df["lag_1"] = engagement.shift(1)
    df["lag_7"] = engagement.shift(7)
    df["rolling_7"] = engagement.transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).mean()
    )
    df["rolling_std_7"] = engagement.transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).std(ddof=0)
    )

    scale = df["rolling_7"].clip(lower=1)
    df["lag_1_ratio"] = df["lag_1"] / scale
    df["lag_7_ratio"] = df["lag_7"] / scale
    df["rolling_std_ratio"] = df["rolling_std_7"] / scale
    df["recent_trend"] = (df["lag_1"] + 1) / (df["lag_7"] + 1)
    df["engagement_ratio"] = df["engagement"] / scale

    return df
