"""Data loading, downsampling, and preprocessing for the fraud model.

Dataset: Bank Account Fraud (BAF) "Base" variant, a synthetic tabular bank
account opening fraud dataset (~1M rows, 30 features, binary target
`fraud_bool`, ~1.1% fraud prevalence). The raw CSV lives outside version
control under `Bank Account Fraud Dataset Suite (NeurIPS 2022)/Base.csv`.

The full dataset is downsampled to a fixed sample size (stratified by the
fraud label) to keep local training/iteration fast, since model accuracy is
not the focus of this project. Splits are temporal (by the `month` column,
0-7) rather than random, matching how the dataset is intended to be used:
earlier months for training, later months held out as validation/test,
simulating deployment on data that arrives after the training window.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "Bank Account Fraud Dataset Suite (NeurIPS 2022)" / "Base.csv"

TARGET_COL = "fraud_bool"
MONTH_COL = "month"

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# Months used per split (BAF paper's suggested temporal split: hold out the
# last two months for validation/test since fraud data is naturally ordered
# in time).
TRAIN_MONTHS = list(range(0, 6))
VAL_MONTHS = [6]
TEST_MONTHS = [7]

SAMPLE_SIZE = 250_000
RANDOM_STATE = 42


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def downsample(df: pd.DataFrame, n: int = SAMPLE_SIZE, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Stratified downsample by fraud label, preserving the original fraud rate."""
    frac = n / len(df)
    parts = [group.sample(frac=frac, random_state=random_state) for _, group in df.groupby(TARGET_COL)]
    sampled = pd.concat(parts).sample(frac=1, random_state=random_state)  # shuffle
    return sampled.reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categoricals, keep `-1` sentinel values as-is (XGBoost
    handles them as ordinary numeric values, and the tree model can learn to
    split on the "missing" sentinel directly)."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    y = df.pop(TARGET_COL)
    return df, y


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df[MONTH_COL].isin(TRAIN_MONTHS)].reset_index(drop=True),
        "val": df[df[MONTH_COL].isin(VAL_MONTHS)].reset_index(drop=True),
        "test": df[df[MONTH_COL].isin(TEST_MONTHS)].reset_index(drop=True),
    }


def load_splits(sample_size: int = SAMPLE_SIZE, random_state: int = RANDOM_STATE):
    """Convenience entry point: load raw data, downsample, split, preprocess.

    Returns a dict of {split_name: (X, y)} for "train", "val", "test".
    """
    raw = load_raw_data()
    sampled = downsample(raw, n=sample_size, random_state=random_state)
    splits = temporal_split(sampled)

    # Fit one-hot encoding columns on the full sampled set so train/val/test
    # share the same feature space, then re-split.
    encoded, _ = preprocess(sampled.drop(columns=[]))
    encoded[TARGET_COL] = sampled[TARGET_COL].values
    encoded[MONTH_COL] = sampled[MONTH_COL].values

    result = {}
    for name, months in [("train", TRAIN_MONTHS), ("val", VAL_MONTHS), ("test", TEST_MONTHS)]:
        subset = encoded[encoded[MONTH_COL].isin(months)].reset_index(drop=True)
        y = subset.pop(TARGET_COL)
        X = subset.drop(columns=[MONTH_COL])
        result[name] = (X, y)
    return result


if __name__ == "__main__":
    splits = load_splits()
    for name, (X, y) in splits.items():
        print(f"{name}: {len(X):>7} rows, fraud rate = {y.mean():.4%}")
