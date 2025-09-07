from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
import pandas as pd

from .io import load_neighborhoods
from .geo import ensure_wgs84, compute_area_km2, dissolve_by_district


def ensure_outputs_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_polygons(neighborhoods_path: str | Path) -> gpd.GeoDataFrame:
    gdf = load_neighborhoods(neighborhoods_path)
    gdf = ensure_wgs84(gdf)
    gdf = compute_area_km2(gdf)
    return gdf


def update_summary(new_df: pd.DataFrame, summary_path: str | Path, key_cols: List[str]) -> pd.DataFrame:
    """Idempotently upsert label columns into a summary CSV keyed by key_cols.

    - Reads existing CSV if present.
    - Outer merges on key_cols, preferring new_df's values for overlapping columns.
    - Writes back to summary_path and returns the updated DataFrame.
    """
    p = Path(summary_path)
    if p.exists():
        existing = pd.read_csv(p)
    else:
        existing = pd.DataFrame(columns=key_cols)
    # Merge and prefer new values
    merged = existing.merge(new_df, on=key_cols, how="outer", suffixes=("_old", ""))
    # For overlapping columns with _old suffix, fill from new
    for col in list(merged.columns):
        if col.endswith("_old"):
            base = col[:-4]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[col])
                merged = merged.drop(columns=[col])
    merged.to_csv(p, index=False)
    return merged


def dissolve_to_districts(gdf_nei: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return dissolve_by_district(gdf_nei)


