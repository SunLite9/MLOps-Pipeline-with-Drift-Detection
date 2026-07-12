"""Sends synthetic prediction requests to the serving API — either matching
the training distribution ("normal" traffic) or deliberately shifted away
from it ("drifted") — to test and demo the drift detector's trigger without
waiting for real drift to occur.

Usage:
    python -m src.drift.simulate_drift --n 150                  # normal traffic
    python -m src.drift.simulate_drift --n 150 --drifted         # shifted traffic
"""

import argparse

import pandas as pd
import requests

from src.training.data import TARGET_COL, load_raw_data

API_URL_DEFAULT = "http://localhost:8000/predict"

# Features shifted for the "drifted" scenario, and the multiplicative factor
# applied to each sampled value. Velocity and session-length are behavioral
# signals that would plausibly shift if fraud patterns changed; age is
# included to push values outside the training range entirely.
DRIFT_SHIFTS = {
    "velocity_6h": 4.0,
    "velocity_24h": 4.0,
    "velocity_4w": 3.0,
    "session_length_in_minutes": 0.1,
    "customer_age": 2.5,
}


def _sample_rows(n: int, random_state: int = None) -> pd.DataFrame:
    raw = load_raw_data()
    sample = raw.drop(columns=[TARGET_COL]).sample(n=n, random_state=random_state)
    return sample.reset_index(drop=True)


def _apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, factor in DRIFT_SHIFTS.items():
        if col in df.columns:
            df[col] = df[col] * factor
    return df


def _to_json_safe(row: pd.Series) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def send_requests(n: int, drifted: bool, api_url: str, random_state: int = None) -> dict:
    df = _sample_rows(n, random_state=random_state)
    if drifted:
        df = _apply_drift(df)

    successes, failures = 0, 0
    for _, row in df.iterrows():
        payload = _to_json_safe(row)
        resp = requests.post(api_url, json=payload, timeout=10)
        if resp.status_code == 200:
            successes += 1
        else:
            failures += 1
            print(f"request failed: {resp.status_code} {resp.text[:200]}")

    summary = {"sent": n, "drifted_scenario": drifted, "ok": successes, "failed": failures}
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--drifted", action="store_true")
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    send_requests(args.n, args.drifted, args.api_url, random_state=args.seed)


if __name__ == "__main__":
    main()
