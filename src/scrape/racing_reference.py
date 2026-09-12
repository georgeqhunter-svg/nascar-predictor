"""Racing Reference scraper.

Parses:
  * Season index (/season-stats/{year}/W/) -> list of Cup races for the year.
  * Race page   (/race-results/{slug}/W/)  -> race metadata + finishing order.

Design goals:
  * Fail loudly on schema drift; assert row counts and required columns.
  * Never lose data: raw HTML always cached (via http.fetch) before parsing.
  * Idempotent: same URL always resolves to same parsed structure.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from io import StringIO
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from .http import fetch

log = logging.getLogger(__name__)

BASE = "https://www.racing-reference.info"
CUP_SERIES = "W"  # Racing Reference series code for NASCAR Cup


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class RaceStub:
    """One row from a season index page."""

    season: int
    race_number: int
    date: date
    slug: str  # e.g. "2024_Daytona_500"
    url: str
    track_name: str
    track_url: Optional[str]
    winner_name: str
    winner_car_number: Optional[int]
    cars: Optional[int]
    make: Optional[str]
    track_length_mi: Optional[float]
    surface: Optional[str]  # 'P' paved / 'R' road / 'S' street / 'D' dirt
    laps: Optional[int]
    pole_speed: Optional[float]
    cautions: Optional[int]
    caution_laps: Optional[int]
    avg_speed: Optional[float]
    lead_changes: Optional[int]


@dataclass
class RaceDetail:
    """Full parsed race page."""

    stub: RaceStub
    race_id_short: str  # e.g. "2024-01" (used for sub-pages: pit stops, loop data)
    race_name: str
    race_datetime_text: str  # e.g. "Monday, February 19, 2024"
    track_name_full: str
    track_city_state: Optional[str]
    laps: int
    track_length_mi: float
    total_miles: float
    time_of_race: Optional[str]
    avg_speed_mph: Optional[float]
    pole_speed_mph: Optional[float]
    cautions_count: Optional[int]
    cautions_laps: Optional[int]
    margin_of_victory: Optional[str]
    lead_changes: Optional[int]
    green_flag_passes: Optional[int]
    entries: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- #
# Season index
# --------------------------------------------------------------------------- #
_TRACK_URL_RE = re.compile(r"/tracks/([^/\"']+)")


def _season_url(year: int) -> str:
    return f"{BASE}/season-stats/{year}/{CUP_SERIES}/"


def fetch_season_index(year: int) -> List[RaceStub]:
    """Return the list of Cup races for `year`, in order.

    Racing Reference's season page renders the schedule as an ARIA
    role="table" made of nested divs (not an HTML <table>), so we walk the
    DOM directly and pull cells by their CSS class labels.
    """
    url = _season_url(year)
    html = fetch(url, subdir="season_index")
    soup = BeautifulSoup(html, "lxml")

    table = soup.select_one('div.statisticsDataTable[role="table"]') or soup.select_one(
        'div.data-table[role="table"]'
    )
    if table is None:
        raise RuntimeError(f"No schedule ARIA-table found on {url}")

    rows = table.select('div.table-row[role="row"]')
    if len(rows) < 20:
        raise RuntimeError(
            f"Season {year} only produced {len(rows)} rows — Racing Reference layout may have changed."
        )

    stubs: List[RaceStub] = []
    for row in rows:
        race_num_cell = row.select_one("div.race-number a")
        if race_num_cell is None:
            continue
        try:
            race_number = int(race_num_cell.get_text(strip=True))
        except ValueError:
            continue

        href = race_num_cell.get("href", "").strip()
        if not href:
            continue
        if not href.startswith("http"):
            href = BASE + href
        slug = href.rstrip("/").rsplit("/", 2)[-2]

        date_txt = _cell_text(row, "date")
        race_date = _parse_short_date(date_txt, year) if date_txt else None
        if race_date is None:
            log.warning("season %s race %s: unparseable date %r", year, race_number, date_txt)
            continue

        track_a = row.select_one("div.track a")
        track_name = (track_a.get_text(strip=True) if track_a else _cell_text(row, "track")) or ""
        track_url = None
        if track_a:
            th = (track_a.get("href") or "").strip()
            if th:
                track_url = th if th.startswith("http") else BASE + th

        winners_a = row.select_one("div.winners a")
        winner_name = (winners_a.get_text(strip=True) if winners_a else _cell_text(row, "winners"))

        stubs.append(
            RaceStub(
                season=year,
                race_number=race_number,
                date=race_date,
                slug=slug,
                url=href,
                track_name=track_name,
                track_url=track_url,
                winner_name=winner_name or "",
                winner_car_number=None,  # not exposed as a distinct cell in the div layout
                cars=_to_int(_cell_text(row, "cars")),
                make=(_cell_text(row, "manufacturer") or None),
                track_length_mi=_to_float(_cell_text(row, "len")),
                surface=(_cell_text(row, "sfc") or None),
                laps=_to_int(_cell_text(row, "laps") or _cell_text(row, "laps_num")),
                pole_speed=_to_float(_cell_text(row, "pole")),
                cautions=_to_int(_cell_text(row, "cau")),
                caution_laps=None,  # not in season index; parsed at race level
                avg_speed=_to_float(_cell_text(row, "speed") or _cell_text(row, "avg")),
                lead_changes=_to_int(_cell_text(row, "lc")),
            )
        )
    return stubs


def _cell_text(row: "BeautifulSoup", first_class: str) -> str:
    """Return trimmed text of the cell whose FIRST class token matches `first_class`.

    Racing Reference cells look like <div class="date W" role="cell">02/20/22</div>.
    We match on the leading token so decorators like ' W' or ' no-mobile' don't matter.
    """
    for cell in row.select('div[role="cell"]'):
        classes = cell.get("class") or []
        if classes and classes[0] == first_class:
            return cell.get_text(strip=True)
    return ""


def _to_int(v: object) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(v)))
    except (TypeError, ValueError):
        return None


def _to_float(v: object) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _parse_short_date(txt: str, year: int) -> date:
    """RR season index shows dates as MM/DD/YY."""
    return dateparser.parse(txt, default=dateparser.parse(f"{year}-01-01")).date()


# --------------------------------------------------------------------------- #
# Race detail
# --------------------------------------------------------------------------- #
_RACE_ID_SHORT_RE = re.compile(rf"/(?:pitstops|loopdata|qual-results|entrylist)/(\d{{4}}-\d{{2}})/")
_LAPS_MILE_RE = re.compile(r"(\d+)\s+laps\s+on\s+a\s+([\d.]+)\s+mile", re.IGNORECASE)
_META_RE = {
    "time_of_race": re.compile(r"Time of race:\s*([\d:]+)"),
    "avg_speed": re.compile(r"Average speed:\s*([\d.]+)\s*mph"),
    "pole_speed": re.compile(r"Pole speed:\s*([\d.]+)\s*mph"),
    "cautions": re.compile(r"Cautions:\s*(\d+)\s+for\s+(\d+)\s+laps"),
    "margin": re.compile(r"Margin of victory:\s*([^\n]+?)(?:\s{2,}|\n|\*)"),
    "lead_changes": re.compile(r"Lead changes:\s*(\d+)"),
    "gf_passes": re.compile(r"Green flag passes:\s*([\d,]+)"),
}


def fetch_race_detail(stub: RaceStub) -> RaceDetail:
    """Fetch and parse a single race page."""
    html = fetch(stub.url, subdir="race_detail")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=False)

    race_id_short = _extract_race_id_short(soup, stub)

    laps, length_mi = _extract_laps_and_length(text)
    total_miles = laps * length_mi if laps and length_mi else 0.0

    meta = {}
    for key, pat in _META_RE.items():
        m = pat.search(text)
        if m:
            meta[key] = m.group(1).strip() if key != "cautions" else (m.group(1), m.group(2))

    # Track name / city from the "at {track link}, City, ST" line.
    track_name_full, city_state = _extract_track_line(soup)

    tables = pd.read_html(StringIO(str(soup)))
    entries = _pick_results_table(tables)
    if entries is None:
        raise RuntimeError(f"No results table found for {stub.url}")
    entries = _clean_results(entries)

    return RaceDetail(
        stub=stub,
        race_id_short=race_id_short,
        race_name=stub.slug.replace("_", " "),
        race_datetime_text=_extract_datetime_text(soup),
        track_name_full=track_name_full or stub.track_name,
        track_city_state=city_state,
        laps=laps,
        track_length_mi=length_mi,
        total_miles=total_miles,
        time_of_race=meta.get("time_of_race"),
        avg_speed_mph=_to_float(meta.get("avg_speed")),
        pole_speed_mph=_to_float(meta.get("pole_speed")),
        cautions_count=_to_int(meta.get("cautions", (None, None))[0]) if "cautions" in meta else None,
        cautions_laps=_to_int(meta.get("cautions", (None, None))[1]) if "cautions" in meta else None,
        margin_of_victory=meta.get("margin"),
        lead_changes=_to_int(meta.get("lead_changes")),
        green_flag_passes=_to_int(meta.get("gf_passes", "").replace(",", "") if meta.get("gf_passes") else None),
        entries=entries,
    )


def _extract_race_id_short(soup: BeautifulSoup, stub: RaceStub) -> str:
    """Find the year-NN race id used for sub-pages (pit stops, loop data, etc)."""
    for a in soup.select("a[href]"):
        m = _RACE_ID_SHORT_RE.search(a["href"])
        if m:
            return m.group(1)
    # Fallback: infer from season + race_number.
    return f"{stub.season}-{stub.race_number:02d}"


def _extract_laps_and_length(text: str) -> tuple[int, float]:
    m = _LAPS_MILE_RE.search(text)
    if not m:
        return 0, 0.0
    return int(m.group(1)), float(m.group(2))


def _extract_track_line(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Find the track link and pull the trailing 'City, ST' from the same paragraph."""
    for a in soup.select("a[href*='/tracks/']"):
        name = a.get_text(strip=True)
        if not name:
            continue
        # Take the parent's text, split around the track name, look for ", City, ST" after.
        parent = a.parent
        if parent is None:
            return name, None
        parent_text = " ".join(parent.stripped_strings)
        m = re.search(re.escape(name) + r"\s*,\s*([A-Za-z .]+,\s*[A-Z]{2})", parent_text)
        city = m.group(1).strip() if m else None
        return name, city
    return None, None


def _extract_datetime_text(soup: BeautifulSoup) -> str:
    text = soup.get_text("\n", strip=False)
    m = re.search(r"(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday),\s*[A-Z][a-z]+\s+\d+,\s+\d{4}", text)
    return m.group(0) if m else ""


def _pick_results_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """The results table has a 'Pos' column, a 'Driver' column, and ~40 rows."""
    for t in tables:
        cols_lower = [str(c).lower() for c in t.columns]
        if "pos" in cols_lower and "driver" in cols_lower and t.shape[0] >= 20:
            return t
        # Sometimes header is on second row; check first data row.
        if t.shape[0] > 5 and t.shape[1] >= 10:
            first_row = [str(x).lower() for x in t.iloc[0].tolist()]
            if "pos" in first_row and "driver" in first_row:
                t2 = t.iloc[1:].copy()
                t2.columns = t.iloc[0].tolist()
                return t2
    return None


def _clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the finishing-order dataframe."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop any leading blank columns.
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]

    rename = {
        "Pos": "finish_pos",
        "St": "start_pos",
        "#": "car_number",
        "Driver": "driver",
        "Sponsor / Owner": "sponsor_owner",
        "Car": "make",
        "Laps": "laps_completed",
        "Status": "status",
        "Led": "laps_led",
        "Pts": "points",
        "PPts": "playoff_points",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Keep only rows where finish_pos parses as an integer.
    df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")
    df = df[df["finish_pos"].notna()].copy()
    df["finish_pos"] = df["finish_pos"].astype(int)

    for col in ("start_pos", "car_number", "laps_completed", "laps_led", "points", "playoff_points"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Split "Sponsor (Owner)" -> sponsor / owner cols.
    if "sponsor_owner" in df.columns:
        so = df["sponsor_owner"].astype(str)
        df["sponsor"] = so.str.replace(r"\s*\(.*\)\s*$", "", regex=True).str.strip()
        df["owner"] = so.str.extract(r"\(([^)]+)\)")[0].str.strip()
        df = df.drop(columns=["sponsor_owner"])

    df["status"] = df.get("status", "").astype(str).str.strip().str.lower()
    df["is_dnf"] = ~df["status"].str.contains("running", na=False)

    return df.reset_index(drop=True)
