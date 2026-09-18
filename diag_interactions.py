"""diag_interactions.py - per-track-type predictive strength of pit/restart/tire features.

For every feature column containing pit/restart/tire, compute Spearman rho vs
actual finish position, pooled and per track type. Diagnostic only; changes nothing.
"""
import pandas as pd
import numpy as np
from src.features.build_features import build_features

KEYWORDS = ("pit", "restart", "tire")

def main():
    races = pd.read_parquet("data/processed/races.parquet")
    entries = pd.read_parquet("data/processed/entries.parquet")
    sessions = pd.read_parquet("data/processed/sessions.parquet")
    loopstats = pd.read_parquet("data/processed/loopstats.parquet")
    laptimes = pd.read_parquet("data/processed/laptimes.parquet")

    print("Building features (slow part)...")
    feats = build_features(races, entries, sessions, loopstats=loopstats, laptimes=laptimes)
    df = feats.reset_index()
    meta = entries[["race_id_short", "driver", "finish_pos"]].drop_duplicates()
    if "finish_pos" not in df.columns:
        df = df.merge(meta, on=["race_id_short", "driver"], how="left")

    cols = [c for c in df.columns if any(k in c.lower() for k in KEYWORDS)]
    if not cols:
        print("No pit/restart/tire columns found in features.")
        return
    print(f"Found {len(cols)} candidate columns: {cols}")
    print()

    types = sorted(df["track_type"].dropna().unique())
    header = f"{'feature':28s} {'pooled':>7s} " + " ".join(f"{t[:9]:>9s}" for t in types)
    print(header)
    print("-" * len(header))
    for c in cols:
        if df[c].notna().sum() < 100:
            print(f"{c:28s}   (only {df[c].notna().sum()} non-null, skipped)")
            continue
        pooled = df[c].corr(df["finish_pos"], method="spearman")
        cells = []
        for t in types:
            g = df[df["track_type"] == t]
            r = g[c].corr(g["finish_pos"], method="spearman") if g[c].notna().sum() >= 40 else np.nan
            cells.append(r)
        row = f"{c:28s} {pooled:+7.3f} " + " ".join(
            (f"{r:+9.3f}" if r == r else f"{'--':>9s}") for r in cells
        )
        print(row)

    print()
    print("Reading: rho>0 = higher feature value -> WORSE finish. Look for |rho| that")
    print("varies a lot across types (e.g. strong at short, weak at intermediate).")
    print("Coverage check (non-null share per column):")
    for c in cols:
        print(f"  {c:28s} {df[c].notna().mean()*100:5.1f}%")

if __name__ == "__main__":
    main()
