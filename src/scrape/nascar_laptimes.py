"""NASCAR.com lap-times scraper.

Endpoint: https://cf.nascar.com/cacher/{year}/{series}/{race_id}/lap-times.json
Returns per-driver arrays of lap times, lap speeds, and running positions.
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


def fetch_laptimes(year: int, race_id: int) -> pd.DataFrame:
    """Return one row per (driver, lap) with lap_time, lap_speed, running_pos.

    Empty DataFrame if no data is published (e.g. race not yet run).
    """
    url = f"{BASE}/cacher/{year}/1/{race_id}/lap-times.json"
    try:
        data = _get_json(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (403, 404):
            return pd.DataFrame()
        raise
    laps_by_driver = (data or {}).get("laps") or []
    if not laps_by_driver:
        return pd.DataFrame()

    def _num(v):
        """NASCAR.com sometimes emits numbers with thousand-separators
        (e.g. '1,133.144' or clearly-corrupt values like '99,663.158').
        Strip commas and coerce; return None on failure."""
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return None

    rows = []
    for d in laps_by_driver:
        driver_id = d.get("NASCARDriverID")
        car_number = d.get("Number")
        for lap in d.get("Laps", []):
            rows.append({
                "driver_id": driver_id,
                "car_number": car_number,
                "lap": int(lap.get("Lap", -1)),
                "lap_time": _num(lap.get("LapTime")),
                "lap_speed": _num(lap.get("LapSpeed")),
                "running_pos": int(lap.get("RunningPos") or 0) or None,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["race_id_nascar"] = int(race_id)
    return df
