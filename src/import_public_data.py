"""Build the training table from a licensed, anonymized Instagram dataset."""

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_FILE = BASE_DIR / "data" / "external" / "grammy_instagram_posts.csv"
OUTPUT_FILE = BASE_DIR / "data" / "social9_engagement.csv"
SOURCE_URL = (
    "https://zenodo.org/api/records/18965670/files/"
    "Grammy_IG_posts_v2.csv/content"
)
SOURCE_MD5 = "b8d14a83559371774b9490abd48bd919"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source() -> None:
    SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE_FILE.exists():
        urlretrieve(SOURCE_URL, SOURCE_FILE)
    if _md5(SOURCE_FILE) != SOURCE_MD5:
        raise ValueError("Downloaded public dataset checksum does not match Zenodo")


def build_training_data() -> pd.DataFrame:
    ensure_source()
    posts = pd.read_csv(SOURCE_FILE, sep=";")
    posts["date"] = pd.to_datetime(
        posts["post_time"], format="%Y-%m-%d_%H-%M-%S_UTC", errors="coerce"
    ).dt.normalize()
    for column in ("likes", "comments", "followers"):
        posts[column] = pd.to_numeric(posts[column], errors="coerce").fillna(0)
        posts[column] = posts[column].clip(lower=0)
    posts = posts.dropna(subset=["date", "user"])

    daily_posts = (
        posts.groupby(["user", "date"], as_index=False)
        .agg(
            likes=("likes", "sum"),
            comments=("comments", "sum"),
            followers=("followers", "max"),
            posts_count=("post_time", "count"),
        )
    )

    histories = []
    for user, account in daily_posts.groupby("user", sort=True):
        if len(account) < 8:
            continue
        account = account.set_index("date").sort_index()
        calendar = pd.date_range(account.index.min(), account.index.max(), freq="D")
        account = account.reindex(calendar)
        account["followers"] = account["followers"].ffill().bfill().fillna(0)
        account[["likes", "comments", "posts_count"]] = account[
            ["likes", "comments", "posts_count"]
        ].fillna(0)
        account["account_id"] = f"zenodo_grammy_{user}"
        account["date"] = account.index
        account["shares"] = 0
        account["reach"] = 0
        account["impressions"] = 0
        account["source_type"] = "real_public_cc_by_4_0"
        histories.append(account.reset_index(drop=True))

    columns = [
        "account_id",
        "date",
        "likes",
        "comments",
        "shares",
        "reach",
        "impressions",
        "followers",
        "posts_count",
        "source_type",
    ]
    result = pd.concat(histories, ignore_index=True)[columns]
    numeric = [
        "likes",
        "comments",
        "shares",
        "reach",
        "impressions",
        "followers",
        "posts_count",
    ]
    result[numeric] = result[numeric].round().astype("int64")
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result.sort_values(["account_id", "date"]).reset_index(drop=True)


def main() -> None:
    dataset = build_training_data()
    dataset.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Built {len(dataset)} real account-day records across "
        f"{dataset['account_id'].nunique()} anonymized Instagram accounts"
    )


if __name__ == "__main__":
    main()
