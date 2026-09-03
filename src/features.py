def create_features(df):

    df = df.copy()

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["date"].dt.day
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["lag_1"] = (
        df["engagement"].shift(1)
    )

    df["lag_7"] = (
        df["engagement"].shift(7)
    )

    df["rolling_7"] = (
        df["engagement"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    return df