import pandas as pd


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
]


def load_data_from_api(
    data: list[dict],
) -> pd.DataFrame:

    if not data:
        raise ValueError(
            "Historical data cannot be empty."
        )

    df = pd.DataFrame(data)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing API fields: {missing}"
        )

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Invalid date found in historical data."
        )

    # Convert numeric fields
    numeric_columns = [
        "likes",
        "comments",
        "shares",
        "reach",
        "impressions",
        "followers",
        "posts_count",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Invalid numeric values found in historical data."
        )

    # Clean Instagram ID
    df["instagram_id"] = (
        df["instagram_id"]
        .astype(str)
        .str.strip()
    )

    # Sort data
    df = df.sort_values(
        ["instagram_id", "date"]
    ).reset_index(drop=True)

    return df