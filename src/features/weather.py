"""Weather features per race.

Historical: Open-Meteo Historical Weather API (free, no key required):
  https://open-meteo.com/en/docs/historical-weather-api
Forecast (upcoming races): Open-Meteo Forecast API:
  https://open-meteo.com/en/docs

We fetch daily aggregates at the track's lat/lon on race date:
  - temperature max (°F)
  - wind speed max (mph)
  - relative humidity mean (%)
  - precipitation sum (in)

Rolling weather-conditioned features per driver:
  - avg finish in hot races (temp_max >= 85 F) — last 20 career races
  - avg finish in cool races (temp_max <= 70 F) — last 20 career races
  - avg finish in windy races (wind_max >= 15 mph) — last 20 career races
  - avg finish in wet races (precip_sum > 0.05 in) — last 20 career races
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from .tracks import TRACKS


CACHE = Path("data/processed/weather.parquet")


# ------------------------------- fetching -------------------------------

def _open_meteo_daily(
    lat: float, lon: float, start: str, end: str, api_kind: str = "archive"
) -> pd.DataFrame:
    """Call Open-Meteo daily endpoint. api_kind is 'archive' (historical) or
    'forecast'. Returns a DataFrame with one row per date.
    """
    import requests

    if api_kind == "archive":
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join([
            "temperature_2m_max",
            "windspeed_10m_max",
            "relative_humidity_2m_mean",
            "precipitation_sum",
        ]),
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    d = j.get("daily", {})
    if not d:
        return pd.DataFrame()
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "temp_max_f": d["temperature_2m_max"],
        "wind_max_mph": d["windspeed_10m_max"],
        "humidity_mean_pct": d["relative_humidity_2m_mean"],
        "precip_sum_in": d["precipitation_sum"],
    })


def fetch_race_weather(races: pd.DataFrame, sleep_s: float = 0.3) -> pd.DataFrame:
    """Fetch daily weather for every race. Match races to tracks by name.

    Args:
      races: races.parquet DataFrame (needs race_id_short, date, track_name).
      sleep_s: pause between requests.

    Returns DataFrame with columns:
      race_id_short, temp_max_f, wind_max_mph, humidity_mean_pct, precip_sum_in
    """
    races = races.copy()
    races["date"] = pd.to_datetime(races["date"])
    today = pd.Timestamp.now().normalize()

    rows = []
    for _, r in races.iterrows():
        rid = r["race_id_short"]
        d = r["date"].strftime("%Y-%m-%d")
        # Match to a track for coords.
        track_name = str(r.get("track_name", ""))
        track = None
        for slug, t in TRACKS.items():
            if slug.replace("_", " ").lower() == track_name.lower():
                track = t
                break
        if track is None:
            rows.append({"race_id_short": rid, "temp_max_f": np.nan,
                         "wind_max_mph": np.nan, "humidity_mean_pct": np.nan,
                         "precip_sum_in": np.nan})
            continue
        api = "forecast" if r["date"] >= today - pd.Timedelta(days=1) else "archive"
        try:
            df = _open_meteo_daily(track.lat, track.lon, d, d, api_kind=api)
            if df.empty:
                raise RuntimeError("empty")
            row = df.iloc[0]
            rows.append({
                "race_id_short": rid,
                "temp_max_f": float(row["temp_max_f"]) if pd.notna(row["temp_max_f"]) else np.nan,
                "wind_max_mph": float(row["wind_max_mph"]) if pd.notna(row["wind_max_mph"]) else np.nan,
                "humidity_mean_pct": float(row["humidity_mean_pct"]) if pd.notna(row["humidity_mean_pct"]) else np.nan,
                "precip_sum_in": float(row["precip_sum_in"]) if pd.notna(row["precip_sum_in"]) else np.nan,
            })
        except Exception as e:
            print(f"  {rid} @ {track.display} {d}: {e}")
            rows.append({"race_id_short": rid, "temp_max_f": np.nan,
                         "wind_max_mph": np.nan, "humidity_mean_pct": np.nan,
                         "precip_sum_in": np.nan})
        time.sleep(sleep_s)
    return pd.DataFrame(rows)


def load_or_fetch_weather(races: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Cached loader. Fetches only rows not already in the parquet cache."""
    if CACHE.exists() and not refresh:
        cached = pd.read_parquet(CACHE)
    else:
        cached = pd.DataFrame(columns=[
            "race_id_short", "temp_max_f", "wind_max_mph",
            "humidity_mean_pct", "precip_sum_in",
        ])
    have = set(cached["race_id_short"].tolist())
    missing = races[~races["race_id_short"].isin(have)]
    if not missing.empty:
        print(f"Fetching weather for {len(missing)} races...")
        new_rows = fetch_race_weather(missing)
        cached = pd.concat([cached, new_rows], ignore_index=True)
        cached.to_parquet(CACHE, index=False)
        print(f"Cached to {CACHE}")
    return cached


# ------------------------- feature computation -------------------------

def compute_weather_features(
    entries: pd.DataFrame,
    races: pd.DataFrame,
    weather: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Per-driver rolling weather-conditioned features (all pre-race).

    Returns one row per (race_id_short, driver) with:
      race_temp_max_f, race_wind_max_mph, race_humidity_pct, race_precip_in
      wx_hot_avg_finish, wx_hot_races
      wx_cool_avg_finish, wx_cool_races
      wx_windy_avg_finish, wx_windy_races
      wx_wet_avg_finish, wx_wet_races
    """
    e = entries.copy()
    e["date"] = pd.to_datetime(e["date"])
    # Merge weather into entries by race.
    e = e.merge(weather, on="race_id_short", how="left")
    e = e.sort_values(["driver", "date"]).reset_index(drop=True)

    HOT = 85.0
    COOL = 70.0
    WIND = 15.0
    WET = 0.05

    e["hot"] = (e["temp_max_f"] >= HOT).astype(int)
    e["cool"] = (e["temp_max_f"] <= COOL).astype(int)
    e["windy"] = (e["wind_max_mph"] >= WIND).astype(int)
    e["wet"] = (e["precip_sum_in"] > WET).astype(int)

    # Rolling by driver: for each condition, compute avg finish over prior
    # window races where that condition was true.
    out_rows = []
    for drv, sub in e.groupby("driver"):
        sub = sub.reset_index(drop=True)
        for i in range(len(sub)):
            prior = sub.iloc[:i].tail(window * 4)  # look back generously
            row = sub.iloc[i]
            def cond_avg(mask_col):
                subp = prior[(prior[mask_col] == 1) & (prior["finish_pos"] > 0)]
                if len(subp) < 2:
                    return np.nan, len(subp)
                return float(subp["finish_pos"].tail(window).mean()), int(min(len(subp), window))
            hot_avg, hot_n = cond_avg("hot")
            cool_avg, cool_n = cond_avg("cool")
            windy_avg, windy_n = cond_avg("windy")
            wet_avg, wet_n = cond_avg("wet")
            out_rows.append({
                "race_id_short": row["race_id_short"],
                "driver": drv,
                "race_temp_max_f": row["temp_max_f"],
                "race_wind_max_mph": row["wind_max_mph"],
                "race_humidity_pct": row["humidity_mean_pct"],
                "race_precip_in": row["precip_sum_in"],
                "wx_hot_avg_finish": hot_avg,
                "wx_hot_races": hot_n,
                "wx_cool_avg_finish": cool_avg,
                "wx_cool_races": cool_n,
                "wx_windy_avg_finish": windy_avg,
                "wx_windy_races": windy_n,
                "wx_wet_avg_finish": wet_avg,
                "wx_wet_races": wet_n,
            })
    return pd.DataFrame(out_rows)
