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
)

from .data_loader import load_data
from .preprocessing import clean_data
from .features import create_features


def train_model():

    print("Loading dataset...")

    df = load_data()

    print(f"Records loaded: {len(df)}")

    # Clean data
    df = clean_data(df)

    # Create Instagram-specific features
    df = create_features(df)

    # Remove rows without lag/rolling values
    df = df.dropna().reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Not enough historical data after feature creation."
        )

    # Sort globally by date for chronological train/test split
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURES]
    y = df["engagement"]

    split_index = int(len(df) * 0.8)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            "Invalid train/test split. Add more historical data."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    print("Training XGBoost model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print()
    print("MODEL PERFORMANCE")
    print("-----------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # Save model
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(str(MODEL_FILE))

    # Save metadata
    metadata = {
        "model": "XGBoost",
        "features": FEATURES,
        "mae": float(mae),
        "rmse": float(rmse),
        "instagram_accounts": int(
            df["instagram_id"].nunique()
        ),
        "training_records": len(X_train),
        "testing_records": len(X_test),
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
    print(f"Model saved: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()