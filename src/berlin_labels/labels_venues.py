from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .text import canon_nh, nationals_set


def percentile_score(series: pd.Series, lo: float = 10.0, hi: float = 90.0) -> pd.Series:
    s = series.astype(float)
    p_lo = np.nanpercentile(s, lo)
    p_hi = np.nanpercentile(s, hi)
    rng = max(p_hi - p_lo, 1e-9)
    scaled = (s - p_lo) / rng
    return (scaled.clip(0, 1) * 100.0)


def tercile_labels(values: pd.Series, labels: Tuple[str, str, str]) -> pd.Series:
    q1 = np.nanpercentile(values, 33.333)
    q2 = np.nanpercentile(values, 66.666)
    def lab(v: float) -> str:
        if np.isnan(v):
            return labels[1]
        if v <= q1:
            return labels[2]
        if v >= q2:
            return labels[0]
        return labels[1]
    return values.apply(lab)


def compute_venues_features(df_venues: pd.DataFrame) -> pd.DataFrame:
    """Compute venue counts and national cuisine diversity per neighborhood.

    Expects columns: district_id, neighborhood, cuisine (semicolon-separated)
    """
    required = {"district_id", "neighborhood", "cuisine"}
    missing = required - set(df_venues.columns)
    if missing:
        raise ValueError(f"df_venues missing columns: {missing}")

    df = df_venues.copy()
    df["neighborhood_canon"] = df["neighborhood"].map(canon_nh)
    # tokenize cuisine to national types
    df["_national_set"] = df["cuisine"].apply(nationals_set)

    # Count venues and number of national cuisine types (diversity)
    grp = df.groupby(["district_id", "neighborhood_canon"], dropna=False)
    n_venues = grp.size().rename("n_venues")
    # union unique types per neighborhood
    unique_types = grp["_national_set"].apply(lambda s: set().union(*s) if len(s) else set()).apply(len)
    unique_types = unique_types.rename("n_cuisine_types")

    out = pd.concat([n_venues, unique_types], axis=1).reset_index()
    return out


def compute_vibrancy_scores(df_features: pd.DataFrame, df_nei_area: pd.DataFrame) -> pd.DataFrame:
    """Compute density-based V_score, C_score and composite VV_index + label.

    df_features must include: district_id, neighborhood_canon, n_venues, n_cuisine_types
    df_nei_area must include: district_id, neighborhood_canon, area_eff_km2, neighborhood/neighborhood_id if needed
    """
    req1 = {"district_id", "neighborhood_canon", "n_venues", "n_cuisine_types"}
    req2 = {"district_id", "neighborhood_canon", "area_eff_km2"}
    if not req1.issubset(df_features.columns):
        raise ValueError("df_features missing required columns")
    if not req2.issubset(df_nei_area.columns):
        raise ValueError("df_nei_area missing required columns")

    df = df_features.merge(df_nei_area, on=["district_id", "neighborhood_canon"], how="left")
    df["venues_per_km2"] = (df["n_venues"].astype(float) / df["area_eff_km2"]).replace([np.inf, -np.inf], np.nan)

    df["V_score"] = percentile_score(df["venues_per_km2"])  # 0–100
    df["C_score"] = percentile_score(df["n_cuisine_types"])  # 0–100
    df["VV_index"] = 0.65 * df["V_score"] + 0.35 * df["C_score"]

    # eligibility
    df["vibrancy_eligible"] = (df["n_venues"] >= 10) & (df["venues_per_km2"] >= 2.0)
    df["vibrancy_label"] = tercile_labels(df["VV_index"], labels=("vibrant", "average", "sparse"))
    # keep label but optionally mark ineligible as sparse if preferred: here we leave labels intact
    return df


