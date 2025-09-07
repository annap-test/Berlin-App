from __future__ import annotations

import numpy as np
import pandas as pd

from .text import canon_nh


def compute_parks_labels(df_parks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate park area and compute green share + label per neighborhood.

    Expects columns: district_id, neighborhood, size_sqm
    """
    required = {"district_id", "neighborhood", "size_sqm"}
    missing = required - set(df_parks.columns)
    if missing:
        raise ValueError(f"df_parks missing columns: {missing}")

    parks = df_parks.copy()
    parks["neighborhood_canon"] = parks["neighborhood"].map(canon_nh)
    parks["green_area_km2"] = parks["size_sqm"].astype(float) / 1e6

    agg = (
        parks.groupby(["district_id", "neighborhood_canon"], dropna=False)["green_area_km2"]
        .sum()
        .reset_index()
    )
    # area_km2 will be merged by caller from polygons; here compute green_share only after merge
    return agg


def label_green_share(df_with_area: pd.DataFrame) -> pd.DataFrame:
    """Compute green_share and label using median band ±0.03."""
    if "green_area_km2" not in df_with_area.columns or "area_km2" not in df_with_area.columns:
        raise ValueError("df_with_area must include green_area_km2 and area_km2")
    df = df_with_area.copy()
    df["green_share"] = (df["green_area_km2"] / df["area_km2"]).replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(df["green_share"]) if len(df) else np.nan
    lower = med - 0.03
    upper = med + 0.03
    def lab(v: float) -> str:
        if np.isnan(v):
            return "average"
        if v < lower:
            return "below average"
        if v > upper:
            return "above average"
        return "average"
    df["green_share_label"] = df["green_share"].apply(lab)
    return df


