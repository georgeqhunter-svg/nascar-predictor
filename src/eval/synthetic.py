"""Synthetic Next Gen NASCAR data generator.

Purpose: exercise the whole model pipeline end-to-end before the real backfill
lands. Ground truth is fully specified so we can measure how well the fitted
ratings recover it.

Schema mirrors what the real scraper produces (races + entries), plus we return
the true latent strengths for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.tracks import TRACKS
from ..models.plackett_luce import TRACK_TYPES


@dataclass
class SyntheticWorld:
    drivers: list[str]
    teams: list[str]
    driver_team: dict[str, str]  # each driver assigned to one team
    true_driver: dict[str, float]
    true_team: dict[str, float]
    true_driver_track: dict[str, dict[str, float]]
    dnf_hazard_by_track_type: dict[str, float]
    races: pd.DataFrame       # season, race_num, date, track_slug, track_type
    entries: pd.DataFrame     # race_id, driver, team, finish_pos, is_dnf


def build_world(
    n_drivers: int = 40,
    n_teams: int = 15,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    races_per_season: int = 36,
    seed: int = 7,
) -> SyntheticWorld:
    rng = np.random.default_rng(seed)

    # --- entities -------------------------------------------------------- #
    drivers = [f"D{i:02d}" for i in range(n_drivers)]
    teams = [f"T{i:02d}" for i in range(n_teams)]

    # Team quality: a few strong teams, most middling.
    true_team = {t: float(v) for t, v in zip(teams, rng.normal(0, 0.5, n_teams))}
    # Give the top 3 teams a real edge (Hendrick/Gibbs/Penske-like).
    top_teams = sorted(teams, key=lambda t: true_team[t], reverse=True)[:3]
    for t in top_teams:
        true_team[t] += 0.6

    # Driver assignment: fill top teams first with best drivers.
    true_driver_raw = rng.normal(0, 0.6, n_drivers)
    true_driver_raw.sort()  # ascending
    # Assign strongest drivers to strongest teams (correlated in real life).
    team_capacities = _team_capacities(n_drivers, n_teams)
    driver_team = {}
    # Sort teams by quality desc; sort drivers by raw skill desc.
    teams_by_q = sorted(teams, key=lambda t: true_team[t], reverse=True)
    drivers_desc = drivers[::-1]  # placeholder order

    # Zip strongest drivers to strongest teams, respecting capacity.
    idx = 0
    for team, cap in zip(teams_by_q, team_capacities):
        for _ in range(cap):
            if idx >= len(drivers):
                break
            driver_team[drivers[idx]] = team
            idx += 1

    # Now attach the noisy driver skills to actual driver ids.
    perm = rng.permutation(n_drivers)
    true_driver = {drivers[i]: float(true_driver_raw[perm[i]]) for i in range(n_drivers)}

    # Per-driver-per-track-type adjustments (specialists).
    true_driver_track: dict[str, dict[str, float]] = {}
    for d in drivers:
        # Most drivers have small track adjustments; a few are specialists.
        specialist_pick = rng.integers(0, 5) == 0
        row = {}
        for tt in TRACK_TYPES:
            adj = rng.normal(0, 0.15)
            if specialist_pick and tt == rng.choice(TRACK_TYPES):
                adj += rng.normal(0.6, 0.2)  # significant specialist bump
            row[tt] = float(adj)
        true_driver_track[d] = row

    # DNF hazard by track type — superspeedways much higher, road courses lowest.
    dnf_hazard_by_track_type = {
        "superspeedway": 0.20,
        "intermediate": 0.05,
        "short": 0.06,
        "road": 0.04,
        "unique": 0.07,
    }

    # --- schedule -------------------------------------------------------- #
    track_slugs = list(TRACKS.keys())
    races_rows = []
    entries_rows = []
    race_num_global = 0
    for season in seasons:
        # Pick a plausible schedule: sample tracks proportional to weights.
        # Just cycle through track_slugs for the first N races.
        for r in range(races_per_season):
            slug = track_slugs[(r + season) % len(track_slugs)]
            tt = TRACKS[slug].track_type
            race_id = f"{season}-{r+1:02d}"
            races_rows.append({
                "race_id_short": race_id,
                "season": season,
                "race_number": r + 1,
                "date": pd.Timestamp(f"{season}-02-01") + pd.Timedelta(days=7 * r),
                "track_slug": slug,
                "track_type": tt,
            })

            # Simulate the finishing order under the true model.
            strengths = np.array([
                true_driver[d] + true_team[driver_team[d]] + true_driver_track[d][tt]
                for d in drivers
            ])
            gumbel = -np.log(-np.log(rng.random(n_drivers).clip(1e-12, 1.0)))
            scores = strengths + gumbel
            order = np.argsort(-scores)  # ascending negative -> best first

            hazard = dnf_hazard_by_track_type[tt]
            dnf_mask = rng.random(n_drivers) < hazard
            # push DNFs to the tail with jittered tail scores
            tail_jitter = rng.random(n_drivers) * 0.01
            scores_dnf = np.where(dnf_mask, -1e6 + tail_jitter, scores)
            order = np.argsort(-scores_dnf)
            for pos, drv_idx in enumerate(order, start=1):
                d = drivers[drv_idx]
                entries_rows.append({
                    "race_id_short": race_id,
                    "season": season,
                    "date": races_rows[-1]["date"],
                    "track_slug": slug,
                    "track_type": tt,
                    "driver": d,
                    "team": driver_team[d],
                    "finish_pos": pos,
                    "is_dnf": bool(dnf_mask[drv_idx]),
                })
            race_num_global += 1

    return SyntheticWorld(
        drivers=drivers,
        teams=teams,
        driver_team=driver_team,
        true_driver=true_driver,
        true_team=true_team,
        true_driver_track=true_driver_track,
        dnf_hazard_by_track_type=dnf_hazard_by_track_type,
        races=pd.DataFrame(races_rows),
        entries=pd.DataFrame(entries_rows),
    )


def _team_capacities(n_drivers: int, n_teams: int) -> list[int]:
    """Distribute drivers across teams — top teams get 4 cars, bottom get 1."""
    base, extra = divmod(n_drivers, n_teams)
    caps = [base + (1 if i < extra else 0) for i in range(n_teams)]
    # Bias: give the strongest few teams a couple extra cars each.
    caps.sort(reverse=True)
    if n_drivers >= n_teams + 2 and caps[-1] > 0:
        # Move 1 car from smallest two teams to top two.
        for from_i, to_i in [(-1, 0), (-2, 1)]:
            if caps[from_i] > 1:
                caps[from_i] -= 1
                caps[to_i] += 1
    return caps
