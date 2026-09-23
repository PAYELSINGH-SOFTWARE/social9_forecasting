def create_features(df):
    df = df.copy()

    # Sort each Instagram account by date
    df = df.sort_values(
        ["instagram_id", "date"]
    ).reset_index(drop=True)

    # Date-based features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Create lag features separately for each Instagram account
    grouped = df.groupby("instagram_id")["engagement"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)

    # Rolling average separately for each account
    df["rolling_7"] = (
        grouped
        .shift(1)
        .groupby(df["instagram_id"])
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df