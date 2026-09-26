"""LightGBM LambdaRank wrapper for ranking drivers within a race.

The model is trained on all races in the training window, using race_id as the
LTR group. The target is a monotonic transformation of finishing position
(higher = better rank; DNQs are already filtered upstream).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "LightGBM is not installed. Run: python -m pip install lightgbm"
    ) from e


# Features the model consumes. Order does not matter for LightGBM but we
# freeze it here so save/load stay consistent.
FEATURES: list[str] = [
    "pl_driver", "pl_team", "pl_driver_track", "pl_effective",
    "qual_z", "practice_z", "start_pos",
    "track_length_mi", "track_banking_deg", "race_distance_mi",
    # `restrictor_plate` (gain 6.6) removed 2026-09-24 — captured by
    # track_length_mi + track_banking_deg + track_type continuous features.
    "avg_finish_5", "avg_finish_10", "avg_finish_20",
    "dnf_rate_10", "top10_rate_10",
    "season_wins_ytd", "season_top5_ytd", "season_top10_ytd",
    "races_at_type_ytd", "avg_finish_at_type_ytd",
    "career_races", "days_since_last_race",
    # Head-to-head beat rates (walk-forward rolling).
    "h2h_beat_rate_10", "h2h_beat_rate_type_10", "h2h_shared_opponents_10",
    # Forward-looking (this-weekend / current-reliability) signals.
    "practice_gap_z", "practice_laps_z",
    "team_teammate_qual_z", "team_teammate_practice_z",
    "mech_dnf_rate_10", "crash_dnf_rate_10", "crash_dnf_rate_at_type_10",
    # Track-type-restricted rolling (spans seasons, not YTD).
    # `races_at_type_last_10` (gain 0.1) removed — no signal, sample-count only.
    "avg_finish_at_type_last_5", "avg_finish_at_type_last_10",
    # Tire-degradation (per-stint pace slope), restricted to same track_type.
    # `tire_races_type` (gain 1.8) removed — sample-count only.
    "tire_decay_type_10", "tire_retention_type_10",
    # Track-specific driver history.
    "races_at_track", "avg_finish_at_track", "best_finish_at_track",
    # Team-track-type rolling.
    "team_avg_finish_at_type", "team_races_at_type",
    # `tm_races_20` (gain 6.5) removed — sample-count only. Keep tm_wpct.
    "tm_wpct_20", "tm_adj_wpct_20",
    # Momentum.
    "momentum_3",
    # Playoff pressure — dropped 2026-09-24 after audit fixes made the
    # is_playoff_driver leak safe but the feature contribution collapsed.
    # `playoff_round` (3.9), `races_to_cutoff` (0.0), `in_playoffs` (0.1),
    # `is_playoff_driver` (1.3), `is_elimination_race` (0.6) all removed.
    # Loop-data rolling features (NaN if driver has no prior loop data yet).
    "loop_avg_ps_5", "loop_avg_ps_10",
    "loop_quality_passes_5", "loop_quality_passes_10",
    "loop_rating_5", "loop_rating_10",
    "loop_fast_laps_10", "loop_passing_diff_10",
    "loop_top15_laps_10", "loop_lead_laps_10",
    # Restart performance (rolling).
    "restart_gain_5", "restart_gain_10", "restart_count_10",
    # Pit-crew performance (rolling, inferred from lap-time patterns).
    "pit_gain_5", "pit_gain_10", "pit_gain_std_10", "pit_time_delta_10",
    # Team-level pit performance — captures crew shuffles + playoff crew
    # assignment better than a specific driver's rolling window would.
    "team_pit_gain_5", "team_pit_gain_10",
    "team_pit_gain_std_5", "team_pit_gain_std_10",
    "team_pit_time_delta_10", "team_pit_n_stops_10",
    "restart_gain_std_10",
    # Driver-specific outcome distribution shape (at track type).
    "dnf_rate_at_type_10", "finish_std_at_type_10", "finish_iqr_at_type_10",
    # Race-pace / manufacturer signals (at track type, walk-forward rolling).
    "laps_led_avg_at_type_10", "laps_led_pct_at_type_10",
    "qual_to_finish_delta_at_type_10",
    "manuf_avg_finish_at_type_10",
    # Track-specific rolling (last 5/10 races AT THIS EXACT TRACK).
    # `races_at_track_last_10` (gain 0.6) removed — sample-count only.
    "avg_finish_at_track_last_5", "avg_finish_at_track_last_10",
    "best_finish_at_track_last_10",
    # LapRaptor practice-pace features (best-lap, best 5/10-lap window, and
    # lap-to-lap consistency from THIS weekend's practice).
    # `has_practice_data` (gain 0.1) removed — dead flag.
    "practice_best_speed_z", "practice_5lap_avg_z", "practice_10lap_avg_z",
    "practice_consistency_z", "practice_laps_run_z",
    # `practice_longrun_gap` tested 2026-09-25, reverted — no Δ gain (5-bp rule).
    # Expanded manufacturer × track-type interactions.
    # `drv_manuf_type_races_10` (gain 6.5) removed — sample-count only.
    "manuf_avg_finish_at_type_5_v2", "manuf_avg_finish_at_type_10_v2",
    "manuf_win_rate_at_type_10", "manuf_top5_rate_at_type_10",
    "manuf_top10_rate_at_type_10", "manuf_finish_std_at_type_10",
    "drv_manuf_type_avg_finish_10",
    # Weather features: this race's conditions.
    "race_temp_max_f", "race_wind_max_mph", "race_humidity_pct", "race_precip_in",
    # Weather features: driver's rolling performance in similar conditions.
    "wx_hot_avg_finish", "wx_hot_races",
    "wx_cool_avg_finish", "wx_cool_races",
    "wx_windy_avg_finish", "wx_windy_races",
    "wx_wet_avg_finish", "wx_wet_races",
    # New signals (2026-09-24) — see src/features/new_signals.py.
    "exp_finish_from_start", "start_stickiness_at_type",
    "green_pace_pct_5", "green_pace_pct_10",
    # `cc_races_together` removed 2026-09-24 — rank ~92/111 by gain, no Δ benefit.
    "cc_avg_finish_10",
    # `similar_track_avg_finish_10` tested 2026-09-25, reverted — Δ +0.0282 ->
    # +0.0293; hurt Darlington (geometry-similar != race-similar).
]

# `track_type` categorical (gain 0.0) removed 2026-09-24 — GBM never split
# on it; continuous track features (length, banking, distance) capture what
# type-bucketing would.
CATEGORICAL: list[str] = ["driver", "team", "manufacturer"]


# Frozen category universe: pandas `.astype("category")` infers categories per
# call, so train and predict end up with different code mappings whenever the
# value sets differ. LightGBM stores splits by code — a driver with code 5 at
# train and code 3 at predict will be routed to the WRONG leaf. Impact is
# most severe for thin-data drivers (back-of-field), whose predictions can
# swing by ~1 std-dev of score. Fix: freeze the category set once, at process
# start, from the union of everything ever seen in entries.parquet, so codes
# are consistent forever.
_FROZEN_CATEGORIES: dict[str, pd.Index] | None = None


def _load_frozen_categories() -> dict[str, pd.Index]:
    """Union of driver / team / manufacturer / track_type across ALL data.
    Cached after first call. Identity is not leakage — we're saying 'this
    string exists,' not 'this driver's performance is X.'"""
    global _FROZEN_CATEGORIES
    if _FROZEN_CATEGORIES is not None:
        return _FROZEN_CATEGORIES
    from pathlib import Path
    proc = Path("data/processed")
    entries = pd.read_parquet(proc / "entries.parquet")
    races = pd.read_parquet(proc / "races.parquet")
    _FROZEN_CATEGORIES = {
        "driver": pd.Index(sorted(entries["driver"].dropna().astype(str).unique())),
        "team": pd.Index(sorted(entries["team"].dropna().astype(str).unique())),
        "manufacturer": pd.Index(
            sorted(entries["make"].dropna().astype(str).unique())
            if "make" in entries.columns else []
        ),
        "track_type": pd.Index(sorted(races["track_type"].dropna().astype(str).unique())),
    }
    return _FROZEN_CATEGORIES


def _prepare_x(df: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
    drop = drop or []
    feats = [f for f in FEATURES if f not in drop]
    cats = [c for c in CATEGORICAL if c not in drop]
    x = df[feats + cats].copy()
    frozen = _load_frozen_categories()
    for c in cats:
        # Use the frozen category list so codes are identical across every
        # fit() and predict_scores() call. Values in x[c] that aren't in the
        # frozen list become NaN, which LightGBM handles as missing (routed
        # to the default direction of each split). That's safer than random
        # re-encoding, and there shouldn't be any unless entries.parquet has
        # been updated between load and predict — in which case the caller
        # should restart the process.
        universe = frozen.get(c)
        if universe is None:
            x[c] = x[c].astype("category")
        else:
            x[c] = pd.Categorical(x[c].astype(str), categories=universe.tolist())
    if "restrictor_plate" in x.columns:
        x["restrictor_plate"] = x["restrictor_plate"].astype(int)
    return x


def _relevance(finish_pos: pd.Series, max_relevance: int = 30) -> np.ndarray:
    """Higher relevance = better finishing position. LambdaRank prefers a
    bounded integer relevance grade; we cap at `max_relevance` for the top
    positions and grade downward."""
    # A 40-car field: winner gets 30, 2nd gets 29, ..., anyone finishing 31+ gets 0.
    rel = np.clip(max_relevance + 1 - finish_pos.astype(int).to_numpy(), 0, max_relevance)
    return rel


@dataclass
class GBMEnsemble:
    """Train N GBMs with different seeds and average their raw scores.

    Averaging raw scores (before Gumbel-max sampling) is equivalent to a
    log-linear vote and consistently reduces log-loss / matchup LL by a
    small amount for free.
    """
    boosters: list["lgb.Booster"] | None = None
    drop_features: list[str] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        n_estimators: int = 3,
        num_boost_round: int = 600,
        num_leaves: int = 31,
        learning_rate: float = 0.04,
        min_data_in_leaf: int = 25,
        lambda_l2: float = 2.0,
        base_seed: int = 0,
        params_override: dict | None = None,
        cat_smooth: float = 50.0,
        cat_l2: float = 20.0,
        min_data_per_group: int = 50,
    ) -> None:
        x = _prepare_x(features, drop=self.drop_features)
        y = _relevance(features["finish_pos"])
        grouped = features.groupby("race_id_short", sort=False).size().tolist()
        cats = [c for c in CATEGORICAL if c not in (self.drop_features or [])]
        boosters = []
        for k in range(n_estimators):
            dtrain = lgb.Dataset(x, label=y, group=grouped,
                                 categorical_feature=cats, free_raw_data=False)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 5, 10],
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "min_data_in_leaf": min_data_in_leaf,
                "lambda_l2": lambda_l2,
                "cat_smooth": cat_smooth,
                "cat_l2": cat_l2,
                "min_data_per_group": min_data_per_group,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "verbosity": -1,
                "seed": base_seed + k,
                "bagging_seed": base_seed + k,
                "feature_fraction_seed": base_seed + k * 7 + 1,
            }
            if params_override:
                params.update(params_override)
            boosters.append(lgb.train(params, dtrain, num_boost_round=num_boost_round))
        self.boosters = boosters

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        assert self.boosters, "call fit() first"
        x = _prepare_x(features, drop=self.drop_features)
        preds = np.stack([b.predict(x) for b in self.boosters], axis=0)
        return preds.mean(axis=0)

    @property
    def booster(self):
        """Backwards-compat: expose first booster for feature_importance readers."""
        return self.boosters[0] if self.boosters else None


@dataclass
class GBMRanker:
    booster: "lgb.Booster | None" = None
    drop_features: list[str] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        num_boost_round: int = 600,
        num_leaves: int = 63,
        learning_rate: float = 0.04,
        min_data_in_leaf: int = 10,
        lambda_l2: float = 0.0,
        random_state: int = 0,
        params_override: dict | None = None,
    ) -> None:
        x = _prepare_x(features, drop=self.drop_features)
        y = _relevance(features["finish_pos"])
        grouped = features.groupby("race_id_short", sort=False).size().tolist()

        cats = [c for c in CATEGORICAL if c not in (self.drop_features or [])]
        dtrain = lgb.Dataset(x, label=y, group=grouped,
                             categorical_feature=cats, free_raw_data=False)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 5, 10],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l2": lambda_l2,
            "verbosity": -1,
            "seed": random_state,
        }
        if params_override:
            params.update(params_override)
        self.booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        assert self.booster is not None, "call fit() first"
        return self.booster.predict(_prepare_x(features, drop=self.drop_features))

    def save(self, path: Path) -> None:
        assert self.booster is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "GBMRanker":
        return cls(booster=lgb.Booster(model_file=str(path)))
