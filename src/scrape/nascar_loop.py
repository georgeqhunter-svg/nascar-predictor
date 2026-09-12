"""NASCAR.com loopstats scraper.

Endpoint: https://cf.nascar.com/loopstats/prod/{year}/1/{race_id}.json
Returns per-driver-per-race telemetry (avg running position, quality passes,
Driver Rating, etc.) — much less noisy than raw finish position and independent
of DNFs.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .nascar_com import _SESSION

log = logging.getLogger(__name__)

BASE = "https://cf.nascar.com"


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _get_json(url: str):
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


LOOP_FIELDS = [
    "driver_id", "start_ps", "mid_ps", "ps", "closing_ps",
    "closing_laps_diff", "best_ps", "worst_ps", "avg_ps",
    "passes_gf", "passing_diff", "passed_gf", "quality_passes",
    "fast_laps", "top15_laps", "lead_laps", "laps", "rating",
]


def fetch_loopstats(year: int, race_id: int) -> pd.DataFrame:
    """Return a DataFrame of per-driver loop stats. Empty if not yet published."""
    url = f"{BASE}/loopstats/prod/{year}/1/{race_id}.json"
    try:
        data = _get_json(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (403, 404):
            return pd.DataFrame()
        raise
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    row = data[0]
    drivers = row.get("drivers") or []
    if not drivers:
        return pd.DataFrame()
    df = pd.DataFrame(drivers)
    df["race_id_nascar"] = int(row.get("race_id", race_id))
    # Ensure all expected columns exist even if endpoint schema changes.
    for c in LOOP_FIELDS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[[*LOOP_FIELDS, "race_id_nascar"]]
