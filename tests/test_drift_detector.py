"""Unit tests for the PSI drift-detection logic. These test the pure
functions directly with synthetic data — no MLflow server, no dataset,
no serving API involved.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift.detector import (
    PSI_THRESHOLD,
    compute_drift,
    compute_training_distribution,
    is_drifted,
)


def _training_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "income": rng.uniform(0, 1, n),
            "customer_age": rng.normal(40, 10, n),
            "velocity_6h": rng.uniform(0, 10000, n),
            "payment_type_AA": rng.integers(0, 2, n),  # simulated one-hot dummy column
        }
    )


def test_compute_training_distribution_excludes_onehot_dummy_columns():
    df = _training_frame()
    distribution = compute_training_distribution(df)
    assert "income" in distribution
    assert "customer_age" in distribution
    assert "payment_type_AA" not in distribution  # excluded: matches a CATEGORICAL_COLS prefix


def test_compute_training_distribution_bin_proportions_sum_to_one():
    df = _training_frame()
    distribution = compute_training_distribution(df)
    for feature, dist in distribution.items():
        assert sum(dist["expected_proportions"]) == pytest.approx(1.0)


def test_identical_distribution_yields_near_zero_psi():
    train_df = _training_frame(n=5000, seed=1)
    distribution = compute_training_distribution(train_df)

    # Live data drawn from the exact same distribution should show ~no drift.
    live_df = _training_frame(n=300, seed=2)
    psi_scores = compute_drift(live_df, distribution)

    drifted, drifted_features = is_drifted(psi_scores, threshold=PSI_THRESHOLD)
    assert not drifted
    assert drifted_features == {}


def test_shifted_distribution_is_flagged_as_drifted():
    train_df = _training_frame(n=5000, seed=1)
    distribution = compute_training_distribution(train_df)

    live_df = _training_frame(n=300, seed=3)
    live_df["velocity_6h"] = live_df["velocity_6h"] * 5  # push far outside the training range

    psi_scores = compute_drift(live_df, distribution)
    drifted, drifted_features = is_drifted(psi_scores, threshold=PSI_THRESHOLD)

    assert drifted
    assert "velocity_6h" in drifted_features
    assert psi_scores["velocity_6h"] > PSI_THRESHOLD


def test_is_drifted_respects_custom_threshold():
    psi_scores = {"feature_a": 0.05, "feature_b": 0.15}
    drifted_strict, features_strict = is_drifted(psi_scores, threshold=0.1)
    drifted_loose, features_loose = is_drifted(psi_scores, threshold=0.5)

    assert drifted_strict and features_strict == {"feature_b": 0.15}
    assert not drifted_loose and features_loose == {}
