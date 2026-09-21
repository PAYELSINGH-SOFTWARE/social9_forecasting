import json

import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from .config import (
    FEATURES,
    MODEL_DIR,
    MODEL_FILE,
    TARGET,
)

from .data_loader import load_data
from .preprocessing import clean_data
from .features import create_features


def train_model():

    print("Loading dataset...")

    df = load_data()

    print(
        f"Records loaded: {len(df)}"
    )

    df = clean_data(df)

    df = create_features(df)

    df = df.dropna().reset_index(
        drop=True
    )

    if len(df) < 10:

        raise ValueError(
            "Not enough historical data."
        )

    X = df[FEATURES]

    y = df[TARGET]

    position = df.groupby("account_id").cumcount()
    account_size = df.groupby("account_id")["account_id"].transform("size")
    test_mask = position >= (account_size * 0.8).astype(int)

    X_train = X.loc[~test_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[~test_mask]

    print(
        f"Training records: {len(X_train)}"
    )

    print(
        f"Testing records: {len(X_test)}"
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    print(
        "Training XGBoost model..."
    )

    model.fit(
        X_train,
        y_train
    )

    ratio_predictions = np.clip(model.predict(X_test), 0, 2.5)
    scale = df.loc[test_mask, "rolling_7"].to_numpy()
    predictions = ratio_predictions * scale
    actual = df.loc[test_mask, "engagement"].to_numpy()
    baseline = df.loc[test_mask, "lag_1"].to_numpy()

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )
    baseline_mae = mean_absolute_error(actual, baseline)
    weighted_absolute_percentage_error = float(
        np.abs(actual - predictions).sum() / max(actual.sum(), 1)
    )
    baseline_weighted_absolute_percentage_error = float(
        np.abs(actual - baseline).sum() / max(actual.sum(), 1)
    )

    print()
    print(
        "MODEL PERFORMANCE"
    )
    print(
        "-----------------"
    )

    print(
        f"MAE  : {mae:.2f}"
    )

    print(
        f"RMSE : {rmse:.2f}"
    )
    print(
        f"Naive lag-1 MAE: {baseline_mae:.2f}"
    )
    print(
        f"WAPE: {weighted_absolute_percentage_error * 100:.2f}% "
        f"(naive: {baseline_weighted_absolute_percentage_error * 100:.2f}%)"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(
        str(MODEL_FILE)
    )

    metadata = {
        "model": "XGBoost",
        "features": FEATURES,
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_mae": float(baseline_mae),
        "wape": weighted_absolute_percentage_error,
        "baseline_wape": baseline_weighted_absolute_percentage_error,
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_test)),
        "dataset_type": "deterministic_synthetic",
    }

    with open(
        MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print()
    print(
        f"Model saved: {MODEL_FILE}"
    )


if __name__ == "__main__":
    train_model()
