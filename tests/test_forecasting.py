from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import create_features


def test_data_loading():

    df = load_data()

    assert not df.empty

    assert "date" in df.columns


def test_preprocessing():

    df = load_data()

    df = clean_data(df)

    assert "engagement" in df.columns

    assert "engagement_rate" in df.columns


def test_features():

    df = load_data()

    df = clean_data(df)

    df = create_features(df)

    assert "lag_1" in df.columns

    assert "lag_7" in df.columns

    assert "rolling_7" in df.columns