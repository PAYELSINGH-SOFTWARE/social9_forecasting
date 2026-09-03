import pandas as pd


def calculate_target(df):

    df = df.copy()

    df["engagement"] = (
        df["likes"]
        + df["comments"]
        + df["shares"]
    )

    return df


def calculate_engagement_rate(df):

    df = df.copy()

    df["engagement_rate"] = 0.0

    mask = df["reach"] > 0

    df.loc[mask, "engagement_rate"] = (
        df.loc[mask, "engagement"]
        / df.loc[mask, "reach"]
    ) * 100

    return df


def clean_data(df):

    df = df.copy()

    df = calculate_target(df)

    df = calculate_engagement_rate(df)

    return df