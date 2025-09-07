from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import wkb, wkt


PathLike = Union[str, Path]


def load_csv(path: PathLike) -> pd.DataFrame:
    return pd.read_csv(path)


def load_neighborhoods(path: PathLike) -> gpd.GeoDataFrame:
    """Load neighborhoods as GeoDataFrame from GeoJSON or CSV with WKB/WKT geometry."""
    p = Path(path)
    if p.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
        gdf = gpd.read_file(p)
        return gdf
    # fallback CSV: expect geometry in WKB or WKT columns
    df = pd.read_csv(p)
    geom_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in {"geometry", "wkb", "wkt", "geom"}:
            geom_col = c
            break
    if geom_col is None:
        raise ValueError("CSV neighborhoods must include a geometry (WKB/WKT) column")
    # Try WKB first, then WKT
    try:
        geom = df[geom_col].apply(lambda v: wkb.loads(v, hex=True) if isinstance(v, str) else wkb.loads(v))
    except Exception:
        geom = df[geom_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs=4326)
    return gdf


def save_geodataframe(gdf: gpd.GeoDataFrame, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".geojson", ".json"}:
        gdf.to_file(p, driver="GeoJSON")
    else:
        gdf.to_file(p)


def save_dataframe(df: pd.DataFrame, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


