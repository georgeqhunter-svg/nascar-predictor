"""End-to-end sanity: ratings must recover ground truth and beat a null model
on the synthetic world."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.eval.backtest import backtest, summarize
from src.eval.synthetic import build_world
from src.models import distribution as dist
from src.models.plackett_luce import Ratings, RaceRanking, update_from_race


def test_synthetic_world_shape():
    w = build_world(seasons=(2022,), races_per_season=10)
    assert len(w.drivers) == 40
    assert w.races.shape[0] == 10
    # every race has an entry per driver
    assert w.entries.shape[0] == 10 * 40
    # each race has a unique winner
    winners = w.entries[w.entries["finish_pos"] == 1]
    assert len(winners) == 10


def test_updates_recover_driver_skill_ordering():
    w = build_world(seasons=(2022, 2023), races_per_season=36, seed=11)
    ratings, _ = backtest(w.races, w.entries)
    # Compare fitted driver rating to true skill.
    truth = np.array([w.true_driver[d] for d in w.drivers])
    fitted = np.array([ratings.driver.get(d, 0.0) for d in w.drivers])
    rho = spearmanr(truth, fitted).statistic
    assert rho > 0.6, f"Fitted rating rank correlation with truth too low: {rho:.2f}"


def test_backtest_beats_null_on_matchup_logloss():
    w = build_world(seasons=(2022, 2023), races_per_season=36, seed=3)
    _, df = backtest(w.races, w.entries, burn_in_races=36)
    live = df[~df["is_burn_in"]]
    # Null model: all matchups 50/50 -> log-loss = ln 2 ~= 0.693
    assert live["matchup_logloss"].mean() < 0.693 - 0.02


def test_summary_table_has_all_types():
    w = build_world(seasons=(2022, 2023), races_per_season=36, seed=5)
    _, df = backtest(w.races, w.entries, burn_in_races=36)
    s = summarize(df)
    assert "ALL" in s.index
