import pandas as pd


def calculate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total engagement.
    """

    df = df.copy()

    df["engagement"] = (
        df["likes"]
        + df["comments"]
        + df["shares"]
    )

    return df


def calculate_engagement_rate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate engagement rate based on reach.
    """

    df = df.copy()

    df["engagement_rate"] = 0.0

    mask = df["reach"] > 0

    df.loc[mask, "engagement_rate"] = (
        df.loc[mask, "engagement"]
        / df.loc[mask, "reach"]
    ) * 100

    # Prevent invalid values
    df["engagement_rate"] = (
        df["engagement_rate"]
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
        .clip(lower=0)
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for forecasting.
    """

    df = df.copy()

    # Ensure numeric columns are valid
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
            errors="coerce",
        ).fillna(0)

        df[column] = df[column].clip(lower=0)

    # Calculate target and engagement rate
    df = calculate_target(df)

    df = calculate_engagement_rate(df)

    return df