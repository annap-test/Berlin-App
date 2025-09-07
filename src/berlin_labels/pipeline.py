from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import pandas as pd

from .geo import ensure_wgs84, compute_area_km2
from .io import load_neighborhoods, load_csv, save_dataframe, save_geodataframe
from .labels_mobility import compute_mobility_labels
from .labels_parks import compute_parks_labels, label_green_share
from .labels_playgrounds import compute_playgrounds_labels, label_playgrounds_density
from .labels_venues import compute_venues_features, compute_vibrancy_scores
from .text import canon_nh


@dataclass
class Paths:
    neighborhoods: Path
    ubahn_csv: Path
    bus_tram_csv: Path
    parks_csv: Path
    playgrounds_csv: Path
    venues_csv: Path
    out_dir: Path


READABLE_RENAMES: Dict[str, str] = {
    # keys
    "district_id": "District ID",
    "district": "District",
    "neighborhood_id": "Neighborhood ID",
    "neighborhood": "Neighborhood",
    "area_km2": "Area (km²)",
    # mobility
    "ubahn_stations": "U-Bahn stations",
    "bus_tram_stops": "Bus/Tram stops",
    "total_stops": "Transit stops total",
    "connectivity_density": "Transit density (/km²)",
    "mobility_score": "Mobility score (0–100)",
    "mobility_label": "Mobility label",
    # parks
    "green_area_km2": "Green area (km²)",
    "green_share": "Green share (0–1)",
    "green_share_label": "Green label",
    # playgrounds
    "n_playgrounds": "Playgrounds",
    "playgrounds_per_km2": "Playgrounds density (/km²)",
    "playgrounds_density_label": "Playgrounds label",
    # venues
    "n_venues": "Venues",
    "venues_per_km2": "Venues density (/km²)",
    "n_cuisine_types": "Cuisine types (national)",
    "V_score": "V score (0–100)",
    "C_score": "C score (0–100)",
    "VV_index": "Vibrancy index (0–100)",
    "vibrancy_label": "Vibrancy label",
}


def run_all_preprocessing(paths: Paths, write_intermediate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Run all steps and return (minimal, merged, polygons) dataframes.

    Also writes outputs to `paths.out_dir` if `write_intermediate` is True.
    """
    out = paths.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load polygons and ensure area + canonical name
    gdf_nei = load_neighborhoods(paths.neighborhoods)
    gdf_nei = ensure_wgs84(gdf_nei)
    gdf_nei = compute_area_km2(gdf_nei)
    if "neighborhood" not in gdf_nei.columns:
        raise ValueError("Neighborhood polygons must include 'neighborhood'")
    gdf_nei["neighborhood_canon"] = gdf_nei["neighborhood"].map(canon_nh)
    if write_intermediate:
        save_geodataframe(gdf_nei, out / "neighborhoods.geojson")

    # Mobility
    df_ubahn = load_csv(paths.ubahn_csv)
    df_bus = load_csv(paths.bus_tram_csv)
    mob = compute_mobility_labels(gdf_nei, df_ubahn, df_bus)
    if write_intermediate:
        save_dataframe(mob, out / "mobility_labels.csv")

    # Parks
    parks_raw = load_csv(paths.parks_csv)
    parks_agg = compute_parks_labels(parks_raw)
    parks = parks_agg.merge(
        gdf_nei[["district_id", "neighborhood_canon", "area_km2"]],
        on=["district_id", "neighborhood_canon"],
        how="left",
    )
    parks = label_green_share(parks)
    if write_intermediate:
        save_dataframe(parks, out / "parks_features.csv")

    # Playgrounds
    plays_raw = load_csv(paths.playgrounds_csv)
    plays_cnt = compute_playgrounds_labels(plays_raw)
    plays = plays_cnt.merge(
        gdf_nei[["district_id", "neighborhood_canon", "area_eff_km2"]],
        on=["district_id", "neighborhood_canon"],
        how="left",
    )
    plays = label_playgrounds_density(plays)
    if write_intermediate:
        save_dataframe(plays, out / "playgrounds_features.csv")

    # Venues
    venues_raw = load_csv(paths.venues_csv)
    venF = compute_venues_features(venues_raw)
    venV = compute_vibrancy_scores(venF, gdf_nei[["district_id", "neighborhood_canon", "area_eff_km2"]])
    if write_intermediate:
        save_dataframe(venF, out / "features_venues_nationals.csv")
        save_dataframe(venV, out / "features_venues_vibrancy.csv")

    # Merge minimal and merged
    base = gdf_nei[[
        "district_id",
        "district" if "district" in gdf_nei.columns else "district_id",
        "neighborhood_id",
        "neighborhood",
        "neighborhood_canon",
        "area_km2",
    ]].copy()
    base.columns = ["district_id", "district", "neighborhood_id", "neighborhood", "neighborhood_canon", "area_km2"]

    minimal = base.merge(
        mob.drop(columns=[c for c in ["district", "area_eff_km2"] if c in mob.columns]),
        on=["district_id", "neighborhood_id", "neighborhood", "neighborhood_canon"],
        how="left",
    ).merge(
        parks[["district_id", "neighborhood_canon", "green_area_km2", "green_share", "green_share_label"]],
        on=["district_id", "neighborhood_canon"],
        how="left",
    ).merge(
        plays[["district_id", "neighborhood_canon", "n_playgrounds", "playgrounds_per_km2", "playgrounds_density_label"]],
        on=["district_id", "neighborhood_canon"],
        how="left",
    )

    merged = minimal.merge(
        venF, on=["district_id", "neighborhood_canon"], how="left"
    ).merge(
        venV[[
            "district_id",
            "neighborhood_canon",
            "venues_per_km2",
            "n_cuisine_types",
            "V_score",
            "C_score",
            "VV_index",
            "vibrancy_label",
        ]],
        on=["district_id", "neighborhood_canon"],
        how="left",
    )

    # Save canonical for app
    if write_intermediate:
        save_dataframe(minimal.drop(columns=["neighborhood_canon"]), out / "neighborhood_labels_minimal.csv")
        save_dataframe(merged.drop(columns=["neighborhood_canon"]), out / "neighborhood_labels_with_venues.csv")

    return minimal, merged, gdf_nei


def make_readable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with user-friendly column names.

    Keeps original keys (IDs) but renames data columns for readability.
    """
    renamed = df.rename(columns=READABLE_RENAMES)
    # keep a stable key order
    cols = [c for c in [
        READABLE_RENAMES.get("district_id", "district_id"),
        READABLE_RENAMES.get("district", "district"),
        READABLE_RENAMES.get("neighborhood_id", "neighborhood_id"),
        READABLE_RENAMES.get("neighborhood", "neighborhood"),
        READABLE_RENAMES.get("area_km2", "area_km2"),
    ] if c in renamed.columns]
    other = [c for c in renamed.columns if c not in cols]
    return renamed[cols + other]


