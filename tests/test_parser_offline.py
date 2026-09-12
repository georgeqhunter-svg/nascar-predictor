"""Offline parser tests using a captured HTML fixture.

These do NOT hit the network. They exercise the same parse code that fetch_race_detail
would call, using a synthetic HTML page shaped like a real Racing Reference race page.
"""
from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.scrape import racing_reference as rr

FIXTURE = Path(__file__).parent / "fixtures" / "race_page_2024_daytona_500.html"


def _parse_fixture():
    html = FIXTURE.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=False)
    tables = pd.read_html(StringIO(html))
    entries = rr._pick_results_table(tables)
    assert entries is not None, "results table not detected"
    return soup, text, rr._clean_results(entries)


def test_results_shape():
    _, _, entries = _parse_fixture()
    assert entries.shape[0] == 40
    assert set(["finish_pos", "start_pos", "car_number", "driver",
                "make", "laps_completed", "status", "laps_led",
                "points", "playoff_points", "sponsor", "owner",
                "is_dnf"]).issubset(entries.columns)


def test_winner_is_byron():
    _, _, entries = _parse_fixture()
    winner = entries[entries["finish_pos"] == 1].iloc[0]
    assert "Byron" in winner["driver"]
    assert winner["car_number"] == 24
    assert winner["make"] == "Chevrolet"
    assert winner["owner"] == "Rick Hendrick"


def test_dnf_detection():
    _, _, entries = _parse_fixture()
    # 8 crash finishes in the fixture: positions 29-35, 38-40 -> 10 crash rows.
    # Verify our DNF counter picks them up (non-running status).
    dnfs = entries[entries["is_dnf"]]
    assert len(dnfs) == 10
    assert set(dnfs["status"].unique()) == {"crash"}


def test_metadata_regexes():
    _, text, _ = _parse_fixture()
    laps, length = rr._extract_laps_and_length(text)
    assert laps == 200
    assert length == 2.5

    m = rr._META_RE["avg_speed"].search(text)
    assert m and float(m.group(1)) == 157.178

    m = rr._META_RE["cautions"].search(text)
    assert m and (int(m.group(1)), int(m.group(2))) == (5, 20)

    m = rr._META_RE["lead_changes"].search(text)
    assert m and int(m.group(1)) == 41


def test_race_id_short_extracted_from_subpage_link():
    soup, _, _ = _parse_fixture()

    class _Stub:
        season = 2024
        race_number = 1

    assert rr._extract_race_id_short(soup, _Stub()) == "2024-01"


def test_track_line_parse():
    soup, _, _ = _parse_fixture()
    name, city = rr._extract_track_line(soup)
    assert name == "Daytona International Speedway"
    assert city == "Daytona Beach, FL"
