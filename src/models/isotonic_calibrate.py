"""Isotonic calibration for predicted probabilities.

More flexible than temperature scaling: fits a monotonic step function that
maps predicted probability to true probability using validation outcomes.
Handles non-uniform miscalibration (e.g., overconfident on 60-70% picks but
well-calibrated on 90%+ picks).

Apply to any product (matchup, top-5, top-10, winner) as long as we have
(predicted_prob, actual_outcome) pairs from a validation window.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Wraps IsotonicRegression with fallback for thin samples."""

    def __init__(self, min_samples: int = 100):
        self.min_samples = min_samples
        self.iso: IsotonicRegression | None = None
        self.fitted = False

    def fit(self, predicted: np.ndarray, actual: np.ndarray) -> "ProbabilityCalibrator":
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        # Drop NaN pairs.
        mask = np.isfinite(predicted) & np.isfinite(actual)
        predicted, actual = predicted[mask], actual[mask]
        if len(predicted) < self.min_samples:
            self.fitted = False
            return self
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        self.iso.fit(predicted, actual)
        self.fitted = True
        return self

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        if not self.fitted or self.iso is None:
            return np.asarray(predicted, dtype=float)
        return self.iso.transform(np.asarray(predicted, dtype=float))
