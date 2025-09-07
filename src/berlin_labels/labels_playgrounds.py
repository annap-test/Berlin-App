from __future__ import annotations

import numpy as np
import pandas as pd

from .text import canon_nh


def compute_playgrounds_labels(df_playgrounds: pd.DataFrame) -> pd.DataFrame:
    """Count playgrounds and compute density + label per neighborhood.

    Expects columns: district_id, neighborhood, green_area_type
    """
    required = {"district_id", "neighborhood", "green_area_type"}
    missing = required - set(df_playgrounds.columns)
    if missing:
        raise ValueError(f"df_playgrounds missing columns: {missing}")

    df = df_playgrounds.copy()
    df["neighborhood_canon"] = df["neighborhood"].map(canon_nh)
    mask = df["green_area_type"].astype(str).str.lower().str.contains("spielplatz", na=False)
    play = df.loc[mask].copy()
    agg = play.groupby(["district_id", "neighborhood_canon"]).size().rename("n_playgrounds").reset_index()
    return agg


def label_playgrounds_density(df_with_area: pd.DataFrame) -> pd.DataFrame:
    """Compute playground density per km^2 and label by median band ±0.30."""
    if "n_playgrounds" not in df_with_area.columns or "area_eff_km2" not in df_with_area.columns:
        raise ValueError("df_with_area must include n_playgrounds and area_eff_km2")
    df = df_with_area.copy()
    df["playgrounds_per_km2"] = (df["n_playgrounds"].astype(float) / df["area_eff_km2"]).replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(df["playgrounds_per_km2"]) if len(df) else np.nan
    lower = med - 0.30
    upper = med + 0.30
    def lab(v: float) -> str:
        if np.isnan(v):
            return "average"
        if v < lower:
            return "below average"
        if v > upper:
            return "above average"
        return "average"
    df["playgrounds_density_label"] = df["playgrounds_per_km2"].apply(lab)
    return df


