from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .geo import ensure_wgs84, compute_area_km2, to_points_gdf, points_within
from .text import canon_nh


def percentile_score(series: pd.Series, lo: float = 10.0, hi: float = 90.0) -> pd.Series:
    """Scale to 0–100 using p10–p90 caps; handles NaN gracefully."""
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


def compute_mobility_labels(
    gdf_nei: gpd.GeoDataFrame,
    df_ubahn: pd.DataFrame,
    df_bus_tram: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mobility connectivity metrics and labels per neighborhood.

    Returns a DataFrame with keys and columns:
    ubahn_stations, bus_tram_stops, total_stops, mobility_score, mobility_label
    """
    if not {"district_id", "neighborhood_id", "neighborhood"}.issubset(set(gdf_nei.columns)):
        raise ValueError("gdf_nei must include district_id, neighborhood_id, neighborhood")

    gdf_nei = ensure_wgs84(gdf_nei.copy())
    gdf_nei = compute_area_km2(gdf_nei)
    join_cols = ["district_id", "neighborhood_id", "neighborhood"]
    nei_keys = gdf_nei[join_cols + ["area_eff_km2"]]

    # Spatially assign points to neighborhoods
    g_ubahn = to_points_gdf(df_ubahn)
    g_bus = to_points_gdf(df_bus_tram)

    # Create a stable polygon index as an attribute to avoid reliance on sjoin internals
    base_polys = gdf_nei.reset_index().rename(columns={"index": "poly_index"})
    polys = base_polys[["poly_index", "geometry"]]
    pts_ubahn = points_within(polys, g_ubahn)
    pts_bus = points_within(polys, g_bus)

    # Determine the polygon index column name after sjoin
    def _poly_idx_col(df: pd.DataFrame) -> str:
        if "poly_index" in df.columns:
            return "poly_index"
        if "poly_index_nei" in df.columns:
            return "poly_index_nei"
        if "index_right" in df.columns:
            return "index_right"
        # last resort: any column that looks like a right index
        for c in df.columns:
            if c.startswith("index_") or c.endswith("_nei"):
                return c
        raise KeyError("Could not find polygon index column after spatial join")

    col_u = _poly_idx_col(pts_ubahn)
    col_b = _poly_idx_col(pts_bus)

    cnt_ubahn = (
        pts_ubahn.groupby(col_u).size().rename("ubahn_stations").to_frame().reset_index().rename(columns={col_u: "poly_index"})
    )
    cnt_bus = (
        pts_bus.groupby(col_b).size().rename("bus_tram_stops").to_frame().reset_index().rename(columns={col_b: "poly_index"})
    )

    # Map counts back to neighborhood keys using poly_index
    base = base_polys.merge(cnt_ubahn, on="poly_index", how="left").merge(cnt_bus, on="poly_index", how="left")
    out = base[join_cols + ["area_eff_km2", "ubahn_stations", "bus_tram_stops"]].copy()
    out[["ubahn_stations", "bus_tram_stops"]] = out[["ubahn_stations", "bus_tram_stops"]].fillna(0).astype(int)
    out["total_stops"] = out["ubahn_stations"] + out["bus_tram_stops"]

    # Weighted density per km^2
    out["connectivity_density"] = (
        (0.7 * out["ubahn_stations"] + 0.3 * out["bus_tram_stops"]) / out["area_eff_km2"]
    )
    out["mobility_score"] = percentile_score(out["connectivity_density"])  # 0–100
    out["mobility_label"] = tercile_labels(out["mobility_score"], labels=("well-connected", "moderate", "remote"))

    out["neighborhood_canon"] = out["neighborhood"].map(canon_nh)
    return out[[
        "district_id",
        "neighborhood_id",
        "neighborhood",
        "neighborhood_canon",
        "ubahn_stations",
        "bus_tram_stops",
        "total_stops",
        "connectivity_density",
        "mobility_score",
        "mobility_label",
    ]]
