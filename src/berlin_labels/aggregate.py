from __future__ import annotations

from pathlib import Path
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


def _find_regional_stats_path() -> Path | None:
    """Try to locate data/raw/regional_statistics.csv starting from CWD and parent."""
    candidates = [
        Path.cwd() / "data" / "raw" / "regional_statistics.csv",
        Path.cwd().parent / "data" / "raw" / "regional_statistics.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_living_area_2023() -> pd.DataFrame | None:
    """Load living area (km^2) per district_id from regional_statistics.csv (year=2023 or latest).

    living_area_ha = total_area_ha - forest_area_ha - water_area_ha
    living_area_km2 = max(living_area_ha, 0) / 100
    """
    p = _find_regional_stats_path()
    if p is None:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    cols_needed = {"district_id", "year", "total_area_ha", "forest_area_ha", "water_area_ha"}
    if not cols_needed.issubset(df.columns):
        return None
    # choose 2023 if available, else latest year
    years = pd.to_numeric(df["year"], errors="coerce")
    target_year = 2023 if (years == 2023).any() else int(np.nanmax(years))
    sub = df[pd.to_numeric(df["year"], errors="coerce") == target_year].copy()
    for c in ["total_area_ha", "forest_area_ha", "water_area_ha"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub["living_area_ha"] = (sub["total_area_ha"] - sub["forest_area_ha"].fillna(0) - sub["water_area_ha"].fillna(0)).clip(lower=0)
    liv = sub[["district_id", "living_area_ha"]].copy()
    liv["living_area_km2"] = liv["living_area_ha"] / 100.0
    liv["living_area_eff_km2"] = liv["living_area_km2"].clip(lower=0.20)
    return liv[["district_id", "living_area_km2", "living_area_eff_km2"]]


def aggregate_mobility_to_district(nei_labels: pd.DataFrame, gdf_nei: gpd.GeoDataFrame) -> pd.DataFrame:
    req = {"district_id", "neighborhood_id", "ubahn_stations", "bus_tram_stops"}
    if not req.issubset(nei_labels.columns):
        raise ValueError("nei_labels missing required mobility columns")
    # Include S-Bahn if available; fall back to 0 if absent
    has_sbahn = "sbahn_stations" in nei_labels.columns
    group_cols = [c for c in ["ubahn_stations", "sbahn_stations" if has_sbahn else None, "bus_tram_stops", "total_stops"] if c is not None]
    counts = nei_labels.groupby(["district_id"], dropna=False)[group_cols].sum().reset_index()
    area = _sum_area(gdf_nei)
    out = counts.merge(area, on="district_id", how="left")
    # Prefer living area for district-level densities if available
    living = _load_living_area_2023()
    if living is not None:
        out = out.merge(living, on="district_id", how="left")
        area_col = "living_area_eff_km2"
    else:
        area_col = "area_eff_km2"
    rail_sum = out["ubahn_stations"] + (out["sbahn_stations"] if has_sbahn and "sbahn_stations" in out.columns else 0)
    out["connectivity_density"] = (0.7 * rail_sum + 0.3 * out["bus_tram_stops"]) / out[area_col]
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
    # Prefer living area for district-level density if available
    living = _load_living_area_2023()
    area_col = "area_eff_km2"
    if living is not None:
        out = out.merge(living[["district_id", "living_area_eff_km2"]], on="district_id", how="left")
        area_col = "living_area_eff_km2"
    out["playgrounds_per_km2"] = (out["n_playgrounds"] / out[area_col]).replace([np.inf, -np.inf], np.nan)
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
    # Prefer living area for district-level density if available
    living = _load_living_area_2023()
    area_col = "area_eff_km2"
    if living is not None:
        out = out.merge(living[["district_id", "living_area_eff_km2"]], on="district_id", how="left")
        area_col = "living_area_eff_km2"
    out["venues_per_km2"] = (out["n_venues"] / out[area_col]).replace([np.inf, -np.inf], np.nan)
    out["V_score"] = percentile_score(out["venues_per_km2"])  # 0–100
    out["C_score"] = percentile_score(out["n_cuisine_types"])  # 0–100
    out["VV_index"] = 0.65 * out["V_score"] + 0.35 * out["C_score"]
    from .labels_venues import tercile_labels as terciles  # reuse helper
    out["vibrancy_label"] = terciles(out["VV_index"], labels=("vibrant", "average", "sparse"))
    return out
