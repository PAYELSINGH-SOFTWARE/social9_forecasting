import pandas as pd

from .config import DATA_FILE


REQUIRED_COLUMNS = [
    "date",
    "likes",
    "comments",
    "shares",
    "reach",
    "impressions",
    "followers",
    "posts_count",
]


def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if "account_id" not in df.columns:
        df["account_id"] = "account_1"

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

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
        ).fillna(0)

        df[column] = df[column].clip(lower=0)

    df = (
        df.sort_values(["account_id", "date"])
        .drop_duplicates(["account_id", "date"])
        .reset_index(drop=True)
    )

    return df
