import pandas as pd

from .config import DATA_FILE


# =========================
# REQUIRED COLUMNS
# =========================

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


# =========================
# LOAD DATA
# =========================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    # Load CSV
    df = pd.read_csv(DATA_FILE)

    # Check missing columns
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # =========================
    # CLEAN INSTAGRAM ID
    # =========================

    df["instagram_id"] = (
        df["instagram_id"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["instagram_id"].notna()
        & (df["instagram_id"] != "")
        & (df["instagram_id"] != "nan")
    ]

    # =========================
    # CLEAN DATE
    # =========================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    # =========================
    # NUMERIC COLUMNS
    # =========================

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

        df[column] = df[column].clip(
            lower=0
        )

    # =========================
    # SORT DATA
    # =========================

    df = (
        df.sort_values(
            ["instagram_id", "date"]
        )
        .drop_duplicates(
            subset=["instagram_id", "date"]
        )
        .reset_index(
            drop=True
        )
    )

    return df