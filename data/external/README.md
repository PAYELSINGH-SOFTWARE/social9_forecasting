# Public training source

`grammy_instagram_posts.csv` is the dataset **Grammy-awarded music artists
Instagram dataset for engagement research**, published on Zenodo under the
Creative Commons Attribution 4.0 license.

- DOI: https://doi.org/10.5281/zenodo.18965670
- Source record: https://zenodo.org/records/18965670
- Source file: `Grammy_IG_posts_v2.csv`
- Source MD5: `b8d14a83559371774b9490abd48bd919`
- Creators: Andrea Pérez López, Tomás Baviera, and Marta Fernández-Diego,
  Universitat Politècnica de València
- License: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Social9 uses the anonymized account identifier, post timestamp, follower count,
likes, and comments. The import pipeline aggregates posts into daily account
histories. Days without a published post are represented with zero post
engagement. The source does not provide shares, reach, or impressions, so those
columns remain zero and are not model features.

Social9 modified the source by parsing timestamps, grouping multiple posts by
anonymized account and publication day, filling non-posting calendar days, and
mapping compatible values into the forecasting training schema. The original
creators do not endorse Social9 or these derived forecasts.
