"""Generate a deterministic, diverse synthetic dataset for model development.

This dataset is intentionally labeled synthetic. It improves engineering and edge-case
coverage but must eventually be replaced or supplemented with consented, anonymized
production histories before forecasting claims are made.
"""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_engagement.csv"
SEED = 20260921
ACCOUNT_COUNT = 16
DAYS_PER_ACCOUNT = 450


def generate_dataset() -> pd.DataFrame:
    random = np.random.default_rng(SEED)
    start = pd.Timestamp("2024-01-01")
    base_levels = [3, 6, 12, 20, 35, 55, 85, 130, 190, 270, 380, 520, 700, 900, 1200, 1600]
    rows: list[dict] = []

    for index, base in enumerate(base_levels):
        account_id = f"synthetic_{index + 1:02d}"
        trend_rate = -0.15 + (index % 6) * 0.18
        volatility = 0.06 + (index % 5) * 0.025
        cadence = 2 + index % 5
        followers_start = max(40, int(base * (8 + index % 7)))
        engagement_rate = 0.025 + (index % 6) * 0.012

        for day in range(DAYS_PER_ACCOUNT):
            current_date = start + pd.Timedelta(days=day)
            weekday = current_date.dayofweek
            weekly = [0.90, 0.98, 1.05, 1.10, 1.16, 1.08, 0.88][
                (weekday + index) % 7
            ]
            annual = 1 + 0.10 * np.sin((2 * np.pi * day / 365) + index / 3)
            trend = max(0.55, 1 + trend_rate * day / DAYS_PER_ACCOUNT)
            campaign_cycle = 31 + index % 17
            campaign = 1.45 if day % campaign_cycle in (0, 1) else 1.0
            posts_count = 1 + int((day + index) % 7 < cadence)
            if (day + index * 3) % 29 == 0:
                posts_count = 0
            posting_effect = 0.72 if posts_count == 0 else 0.92 + posts_count * 0.08
            noise = max(0.45, random.normal(1.0, volatility))
            engagement = max(
                0,
                int(round(base * weekly * annual * trend * campaign * posting_effect * noise)),
            )

            comment_share = min(0.22, max(0.08, random.normal(0.14, 0.02)))
            share_share = min(0.15, max(0.03, random.normal(0.07, 0.015)))
            comments = int(round(engagement * comment_share))
            shares = int(round(engagement * share_share))
            likes = max(0, engagement - comments - shares)
            reach = 0 if engagement == 0 else int(round(engagement / engagement_rate))
            impressions = int(round(reach * (1.15 + (index % 4) * 0.08)))
            followers = followers_start + int(day * max(0.2, base / 180))

            rows.append(
                {
                    "account_id": account_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "reach": reach,
                    "impressions": impressions,
                    "followers": followers,
                    "posts_count": posts_count,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    dataset = generate_dataset()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT, index=False)
    print(
        f"Generated {len(dataset)} synthetic records across "
        f"{dataset['account_id'].nunique()} accounts at {OUTPUT}"
    )


if __name__ == "__main__":
    main()
