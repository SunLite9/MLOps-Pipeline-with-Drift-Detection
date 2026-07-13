"""Concurrent load test against the serving API's /predict endpoint.

The latency numbers elsewhere in this project's docs (avg ~11ms, p99 ~18ms)
were measured from ~600 sequential single-client requests sent by
src/drift/simulate_drift.py — real, but not a load test: sequential
requests never exercise queuing, contention on the model-state lock, or
GIL/thread-pool behavior under concurrency. This script sends genuinely
concurrent requests and reports latency percentiles + throughput.

Usage:
    python -m scripts.load_test --concurrency 20 --requests 500
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_URL_DEFAULT = "http://localhost:8000/predict"

SAMPLE_REQUEST = {
    "income": 0.3,
    "name_email_similarity": 0.98,
    "prev_address_months_count": -1,
    "current_address_months_count": 25,
    "customer_age": 40,
    "days_since_request": 0.006,
    "intended_balcon_amount": 102.45,
    "payment_type": "AA",
    "zip_count_4w": 1059,
    "velocity_6h": 13096.0,
    "velocity_24h": 7850.9,
    "velocity_4w": 6742.0,
    "bank_branch_count_8w": 5,
    "date_of_birth_distinct_emails_4w": 5,
    "employment_status": "CB",
    "credit_risk_score": 163,
    "email_is_free": 1,
    "housing_status": "BC",
    "phone_home_valid": 0,
    "phone_mobile_valid": 1,
    "bank_months_count": 9,
    "has_other_cards": 0,
    "proposed_credit_limit": 1500.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 16.2,
    "device_os": "linux",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
    "device_fraud_count": 0,
    "month": 0,
}


def _one_request(api_url: str) -> tuple[float, int]:
    start = time.perf_counter()
    resp = requests.post(api_url, json=SAMPLE_REQUEST, timeout=30)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, resp.status_code


def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return sorted_values[idx]


def run_load_test(concurrency: int, n_requests: int, api_url: str) -> dict:
    latencies = []
    status_codes = []

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, api_url) for _ in range(n_requests)]
        for future in as_completed(futures):
            elapsed_ms, status_code = future.result()
            latencies.append(elapsed_ms)
            status_codes.append(status_code)
    wall_elapsed_s = time.perf_counter() - wall_start

    latencies.sort()
    ok_count = sum(1 for s in status_codes if s == 200)

    summary = {
        "concurrency": concurrency,
        "requests": n_requests,
        "ok": ok_count,
        "failed": n_requests - ok_count,
        "wall_seconds": round(wall_elapsed_s, 2),
        "throughput_rps": round(n_requests / wall_elapsed_s, 2),
        "avg_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "p99_ms": round(_percentile(latencies, 0.99), 2),
        "max_ms": round(max(latencies), 2),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    args = parser.parse_args()

    summary = run_load_test(args.concurrency, args.requests, args.api_url)
    for k, v in summary.items():
        print(f"{k:>15}: {v}")


if __name__ == "__main__":
    main()
