from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from .labels_mobility import percentile_score, tercile_labels
from .text import canon_nh, nationals_set


def _sum_area(df_nei: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [c for c in ["district_id", "district", "area_km2", "area_eff_km2"] if c in df_nei.columns]
    gg = df_nei[cols].copy()
    grp = gg.groupby([c for c in ["district_id", "district"] if c in gg.columns], dropna=False).sum(numeric_only=True).reset_index()
    if "area_eff_km2" not in grp.columns and "area_km2" in grp.columns:
        grp["area_eff_km2"] = grp["area_km2"].clip(lower=0.20)
    return grp


def aggregate_mobility_to_district(nei_labels: pd.DataFrame, gdf_nei: gpd.GeoDataFrame) -> pd.DataFrame:
    req = {"district_id", "neighborhood_id", "ubahn_stations", "bus_tram_stops"}
    if not req.issubset(nei_labels.columns):
        raise ValueError("nei_labels missing required mobility columns")
    counts = nei_labels.groupby(["district_id"], dropna=False)[["ubahn_stations", "bus_tram_stops", "total_stops"]].sum().reset_index()
    area = _sum_area(gdf_nei)
    out = counts.merge(area, on="district_id", how="left")
    out["connectivity_density"] = (0.7 * out["ubahn_stations"] + 0.3 * out["bus_tram_stops"]) / out["area_eff_km2"]
    out["mobility_score"] = percentile_score(out["connectivity_density"])  # 0–100
    out["mobility_label"] = tercile_labels(out["mobility_score"], labels=("well-connected", "moderate", "remote"))
    return out


def aggregate_parks_to_district(parks_nei: pd.DataFrame, gdf_nei: gpd.GeoDataFrame) -> pd.DataFrame:
    req = {"district_id", "green_area_km2"}
    if not req.issubset(parks_nei.columns):
        raise ValueError("parks_nei missing required columns")
    agg = parks_nei.groupby(["district_id"], dropna=False)["green_area_km2"].sum().reset_index()
    area = _sum_area(gdf_nei)[["district_id", "area_km2"]]
    out = agg.merge(area, on="district_id", how="left")
    out["green_share"] = (out["green_area_km2"] / out["area_km2"]).replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(out["green_share"]) if len(out) else np.nan
    lower, upper = med - 0.03, med + 0.03
    def lab(v: float) -> str:
        if np.isnan(v):
            return "average"
        if v < lower:
            return "below average"
        if v > upper:
            return "above average"
        return "average"
    out["green_share_label"] = out["green_share"].apply(lab)
    return out


def aggregate_playgrounds_to_district(play_nei: pd.DataFrame, gdf_nei: gpd.GeoDataFrame) -> pd.DataFrame:
    req = {"district_id", "n_playgrounds"}
    if not req.issubset(play_nei.columns):
        raise ValueError("play_nei missing required columns")
    agg = play_nei.groupby(["district_id"], dropna=False)["n_playgrounds"].sum().reset_index()
    area = _sum_area(gdf_nei)[["district_id", "area_eff_km2"]]
    out = agg.merge(area, on="district_id", how="left")
    out["playgrounds_per_km2"] = (out["n_playgrounds"] / out["area_eff_km2"]).replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(out["playgrounds_per_km2"]) if len(out) else np.nan
    lower, upper = med - 0.30, med + 0.30
    def lab(v: float) -> str:
        if np.isnan(v):
            return "average"
        if v < lower:
            return "below average"
        if v > upper:
            return "above average"
        return "average"
    out["playgrounds_density_label"] = out["playgrounds_per_km2"].apply(lab)
    return out


def aggregate_venues_to_district(venues_raw: pd.DataFrame, gdf_nei: gpd.GeoDataFrame) -> pd.DataFrame:
    # Expect district_id + cuisine; if neighborhood available, it is ignored
    if "district_id" not in venues_raw.columns or "cuisine" not in venues_raw.columns:
        raise ValueError("venues_raw must include district_id and cuisine")
    df = venues_raw.copy()
    grp = df.groupby("district_id")
    n_venues = grp.size().rename("n_venues")
    unique_types = grp["cuisine"].apply(lambda s: len(set().union(*(nationals_set(v) for v in s if pd.notna(v)))))
    unique_types = unique_types.rename("n_cuisine_types")
    out = pd.concat([n_venues, unique_types], axis=1).reset_index()
    area = _sum_area(gdf_nei)[["district_id", "area_eff_km2"]]
    out = out.merge(area, on="district_id", how="left")
    out["venues_per_km2"] = (out["n_venues"] / out["area_eff_km2"]).replace([np.inf, -np.inf], np.nan)
    out["V_score"] = percentile_score(out["venues_per_km2"])  # 0–100
    out["C_score"] = percentile_score(out["n_cuisine_types"])  # 0–100
    out["VV_index"] = 0.65 * out["V_score"] + 0.35 * out["C_score"]
    from .labels_venues import tercile_labels as terciles  # reuse helper
    out["vibrancy_label"] = terciles(out["VV_index"], labels=("vibrant", "average", "sparse"))
    return out


