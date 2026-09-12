"""NASCAR.com Fantasy Live / cacher-endpoints scraper.

These are the official NASCAR data endpoints powering fantasy.nascar.com. They
return JSON, are not Cloudflare-protected, and include richer fields than
Racing Reference (crew chief, qualifying position, playoff points, etc).

Two endpoints:
  * https://cf.nascar.com/cacher/{year}/{series}/race_list_basic.json
      -> list of all races for a season, with internal `race_id`.
  * https://cf.nascar.com/cacher/{year}/{series}/{race_id}/weekend-feed.json
      -> {"weekend_race": [race_metadata], "results": [per_driver_row]}

Series codes: 1=Cup, 2=Xfinity, 3=Trucks. We use 1.
race_type_id: 1=points, 2=exhibition/Duels/Clash, 3=All-Star. We filter to 1.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, List, Optional

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..features.tracks import TRACKS, TrackType

log = logging.getLogger(__name__)

CUP_SERIES = 1
POINTS_RACE = 1
BASE = "https://cf.nascar.com"

# One shared, throttled session with a realistic UA.
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,*/*;q=0.8",
})


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _get_json(url: str) -> Any:
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class RaceStub:
    race_id: int
    season: int
    race_name: str
    race_type_id: int
    restrictor_plate: bool
    track_id: int
    track_name: str
    race_date: date
    scheduled_laps: int
    actual_laps: int
    stage_1_laps: int
    stage_2_laps: int
    stage_3_laps: int


@dataclass
class RaceDetail:
    stub: RaceStub
    scheduled_distance: float
    actual_distance: float
    number_of_cars_in_field: int
    pole_winner_driver_id: Optional[int]
    pole_winner_speed: Optional[float]
    number_of_lead_changes: int
    number_of_cautions: int
    number_of_caution_laps: int
    average_speed: Optional[float]
    total_race_time: Optional[str]
    margin_of_victory: Optional[str]
    winner_driver_id: Optional[int]
    playoff_round: int
    entries: pd.DataFrame = field(default_factory=pd.DataFrame)
    sessions: pd.DataFrame = field(default_factory=pd.DataFrame)


# NASCAR.com run_type codes for weekend_runs.
RUN_TYPE_PRACTICE = 1
RUN_TYPE_QUALIFYING = 2
RUN_TYPE_LABELS = {1: "practice", 2: "qualifying", 3: "race"}


# --------------------------------------------------------------------------- #
# Season index
# --------------------------------------------------------------------------- #
def _season_url(year: int, series: int = CUP_SERIES) -> str:
    return f"{BASE}/cacher/{year}/{series}/race_list_basic.json"


def fetch_season_index(year: int, points_only: bool = True) -> List[RaceStub]:
    """List all Cup races for `year`. If points_only, drop exhibition/all-star."""
    url = _season_url(year)
    data = _get_json(url)
    stubs: List[RaceStub] = []
    for row in data:
        if points_only and int(row.get("race_type_id", 0)) != POINTS_RACE:
            continue
        # Some qualifying-race entries also appear; skip them.
        if row.get("is_qualifying_race"):
            continue
        stubs.append(RaceStub(
            race_id=int(row["race_id"]),
            season=int(row["race_season"]),
            race_name=str(row.get("race_name", "")).strip(),
            race_type_id=int(row.get("race_type_id", 0)),
            restrictor_plate=bool(row.get("restrictor_plate", False)),
            track_id=int(row.get("track_id", 0)),
            track_name=str(row.get("track_name", "")).strip(),
            race_date=_parse_iso_date(row.get("race_date") or row.get("date_scheduled")),
            scheduled_laps=int(row.get("scheduled_laps") or 0),
            actual_laps=int(row.get("actual_laps") or 0),
            stage_1_laps=int(row.get("stage_1_laps") or 0),
            stage_2_laps=int(row.get("stage_2_laps") or 0),
            stage_3_laps=int(row.get("stage_3_laps") or 0),
        ))
    stubs.sort(key=lambda s: s.race_date)
    return stubs


# --------------------------------------------------------------------------- #
# Race detail
# --------------------------------------------------------------------------- #
def _race_url(year: int, race_id: int, series: int = CUP_SERIES) -> str:
    return f"{BASE}/cacher/{year}/{series}/{race_id}/weekend-feed.json"


def fetch_race_detail(stub: RaceStub) -> RaceDetail:
    url = _race_url(stub.season, stub.race_id)
    data = _get_json(url)

    races = data.get("weekend_race") or []
    if not races:
        raise RuntimeError(f"No weekend_race payload for race {stub.race_id}")
    meta = races[0]
    # `results` is nested inside weekend_race[0]; top-level `results` is not
    # populated on the cacher endpoint we hit.
    results = meta.get("results") or data.get("results") or []

    entries = _clean_results(results)
    sessions = _clean_sessions(data.get("weekend_runs") or [])
    return RaceDetail(
        stub=stub,
        scheduled_distance=float(meta.get("scheduled_distance") or 0.0),
        actual_distance=float(meta.get("actual_distance") or 0.0),
        number_of_cars_in_field=int(meta.get("number_of_cars_in_field") or 0),
        pole_winner_driver_id=_optional_int(meta.get("pole_winner_driver_id")),
        pole_winner_speed=_optional_float(meta.get("pole_winner_speed")),
        number_of_lead_changes=int(meta.get("number_of_lead_changes") or 0),
        number_of_cautions=int(meta.get("number_of_cautions") or 0),
        number_of_caution_laps=int(meta.get("number_of_caution_laps") or 0),
        average_speed=_optional_float(meta.get("average_speed")),
        total_race_time=meta.get("total_race_time") or None,
        margin_of_victory=meta.get("margin_of_victory") or None,
        winner_driver_id=_optional_int(meta.get("winner_driver_id")),
        playoff_round=int(meta.get("playoff_round") or 0),
        entries=entries,
        sessions=sessions,
    )


# --------------------------------------------------------------------------- #
# Results normalization
# --------------------------------------------------------------------------- #
def _clean_results(results: list[dict]) -> pd.DataFrame:
    """Filter to actual starters/finishers and normalize columns."""
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)

    # NASCAR.com puts DNQs at finishing_position=0 for completed races AND
    # returns finishing_position=0 for ALL drivers when a race hasn't been run
    # yet (pre-race entry list). Distinguish: if any driver has fp > 0, the
    # race has been run and 0 = DNQ so filter. Otherwise keep them as the
    # entry list (used for prediction).
    fp = pd.to_numeric(df["finishing_position"], errors="coerce").fillna(0)
    if (fp > 0).any():
        df = df[fp > 0].copy()
    else:
        df = df.copy()  # pre-race — keep entry list, finish_pos stays 0

    keep = {
        "finishing_position": "finish_pos",
        "starting_position": "start_pos",
        "car_number": "car_number",
        "driver_id": "driver_id",
        "driver_fullname": "driver",
        "team_id": "team_id",
        "team_name": "team",
        "owner_id": "owner_id",
        "owner_fullname": "owner",
        "crew_chief_id": "crew_chief_id",
        "crew_chief_fullname": "crew_chief",
        "car_make": "make",
        "car_model": "car_model",
        "sponsor": "sponsor",
        "qualifying_position": "qual_pos",
        "qualifying_speed": "qual_speed",
        "laps_completed": "laps_completed",
        "laps_led": "laps_led",
        "times_led": "times_led",
        "points_earned": "points",
        "playoff_points_earned": "playoff_points",
        "finishing_status": "status",
        "disqualified": "disqualified",
        "diff_laps": "diff_laps",
        "diff_time": "diff_time",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)

    df["finish_pos"] = df["finish_pos"].astype(int)
    for col in ("start_pos", "team_id", "owner_id", "crew_chief_id", "driver_id",
                "qual_pos", "laps_completed", "laps_led", "times_led",
                "points", "playoff_points", "diff_laps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("qual_speed", "diff_time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["status"] = df["status"].astype(str).str.strip()
    # DNF = anything other than "Running" (Accident, Engine, Suspension, Brakes, ...).
    df["is_dnf"] = ~df["status"].str.lower().str.startswith("running")

    return df.sort_values("finish_pos").reset_index(drop=True)


def _clean_sessions(runs: list[dict]) -> pd.DataFrame:
    """Flatten weekend_runs (practice + qualifying) into one long DataFrame.

    Output columns:
        run_id, run_type, run_type_label, run_name, run_date_utc,
        driver_id, driver_name, car_number, manufacturer,
        session_position, best_lap_time, best_lap_speed, best_lap_number,
        laps_completed, delta_leader, disqualified
    """
    rows: list[dict] = []
    for run in runs or []:
        run_id = run.get("weekend_run_id")
        run_type = int(run.get("run_type") or 0)
        run_name = run.get("run_name") or ""
        run_date_utc = run.get("run_date_utc") or run.get("run_date")
        for r in run.get("results") or []:
            rows.append({
                "run_id": run_id,
                "run_type": run_type,
                "run_type_label": RUN_TYPE_LABELS.get(run_type, f"type_{run_type}"),
                "run_name": run_name,
                "run_date_utc": run_date_utc,
                "driver_id": _optional_int(r.get("driver_id")),
                "driver_name": r.get("driver_name") or "",
                "car_number": r.get("car_number") or r.get("vehicle_number") or "",
                "manufacturer": r.get("manufacturer") or "",
                "session_position": _optional_int(r.get("finishing_position")),
                "best_lap_time": _optional_float(r.get("best_lap_time")),
                "best_lap_speed": _optional_float(r.get("best_lap_speed")),
                "best_lap_number": _optional_int(r.get("best_lap_number")),
                "laps_completed": _optional_int(r.get("laps_completed")),
                "delta_leader": _optional_float(r.get("delta_leader")),
                "disqualified": bool(r.get("disqualified", False)),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "run_id", "run_type", "run_type_label", "run_name", "run_date_utc",
            "driver_id", "driver_name", "car_number", "manufacturer",
            "session_position", "best_lap_time", "best_lap_speed",
            "best_lap_number", "laps_completed", "delta_leader", "disqualified",
        ])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Track-type mapping
# --------------------------------------------------------------------------- #
def _norm_track_name(name: str) -> str:
    """Lowercase, strip common suffixes and diacritics."""
    import unicodedata
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()
    # Note: we intentionally do NOT strip " road course" — otherwise
    # "Charlotte Motor Speedway Road Course" collapses onto
    # "Charlotte Motor Speedway" and the two entries collide, mislabeling
    # Coke 600s as road and Verizon 200s as unique.
    for suf in (" (oval)", " (road course)"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


# One-time lookup: normalized NASCAR.com name -> Track.track_type.
_TRACK_LOOKUP: dict[str, TrackType] = {}
for _t in TRACKS.values():
    _TRACK_LOOKUP[_norm_track_name(_t.slug.replace("_", " "))] = _t.track_type
# Explicit aliases where NASCAR.com's naming differs from our slug spelling.
_ALIASES: dict[str, str] = {
    "world wide technology raceway": "world wide technology raceway at gateway",
    "auto club speedway": "auto club speedway",  # not in TRACKS (retired post-2023)
    "los angeles memorial coliseum": "los angeles memorial coliseum",  # exhibition
    "north wilkesboro speedway": "north wilkesboro speedway",  # all-star
    "circuit of the americas": "circuit of the americas",
}


def track_type_for(track_name: str) -> TrackType:
    """Map a NASCAR.com track_name to our TrackType, defaulting to 'intermediate'.

    Unknown tracks (e.g., exhibition venues we don't model) get 'intermediate'
    but should be filtered out upstream by only keeping points races.
    """
    n = _norm_track_name(track_name)
    if n in _TRACK_LOOKUP:
        return _TRACK_LOOKUP[n]
    if n in _ALIASES and _ALIASES[n] in _TRACK_LOOKUP:
        return _TRACK_LOOKUP[_ALIASES[n]]
    log.warning("Unknown track %r — defaulting to 'intermediate'", track_name)
    return "intermediate"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_iso_date(s: Optional[str]) -> date:
    if not s:
        return date(1970, 1, 1)
    # NASCAR.com uses ISO datetime like "2024-02-19T16:00:00".
    return datetime.fromisoformat(s[:19]).date()


def _optional_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _optional_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
