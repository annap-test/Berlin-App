from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in EPSG:4326."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def compute_area_km2(gdf: gpd.GeoDataFrame, area_col: str = "area_km2") -> gpd.GeoDataFrame:
    """Compute area in km^2 using EPSG:25833 (ETRS89 / UTM zone 33N)."""
    g = gdf
    if area_col not in g.columns or g[area_col].isna().any():
        g_utm = ensure_wgs84(g).to_crs(25833)
        areas = g_utm.geometry.area / 1e6
        if area_col in g.columns:
            g[area_col] = g[area_col].fillna(areas)
        else:
            g[area_col] = areas
    # effective area floor to avoid exploding densities
    g["area_eff_km2"] = g[area_col].clip(lower=0.20)
    return g


def _infer_lat_lon_columns(df: pd.DataFrame) -> Optional[tuple[str, str]]:
    cand_lats = [c for c in df.columns if c.lower() in {"lat", "latitude", "y"}]
    cand_lons = [c for c in df.columns if c.lower() in {"lon", "lng", "long", "longitude", "x"}]
    if cand_lats and cand_lons:
        return cand_lats[0], cand_lons[0]
    return None


def to_points_gdf(df: pd.DataFrame, crs: int = 4326) -> gpd.GeoDataFrame:
    """Convert a DataFrame with lat/lon or geometry to a GeoDataFrame (WGS84)."""
    if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
        return ensure_wgs84(df)
    if "geometry" in df.columns and df["geometry"].dtype.name == "geometry":
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
        return ensure_wgs84(gdf)
    ll = _infer_lat_lon_columns(df)
    if ll is None:
        raise ValueError("Cannot infer lat/lon columns; expected 'lat'/'lon' or geometry column")
    lat, lon = ll
    geom = [Point(xy) for xy in zip(df[lon], df[lat])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs=crs)
    return ensure_wgs84(gdf)


def points_within(polygons: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join: annotate points with polygon attributes using within predicate.

    Uses suffixes to avoid name collisions: left (points) = `_pt`, right (polygons) = `_nei`.
    """
    polys = ensure_wgs84(polygons)
    pts = ensure_wgs84(points)
    return gpd.sjoin(pts, polys, how="inner", predicate="within", lsuffix="_pt", rsuffix="_nei")


def dissolve_by_district(gdf_nei: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create district polygons by dissolving neighborhoods on (district_id, district)."""
    cols = [c for c in ["district_id", "district"] if c in gdf_nei.columns]
    if not cols:
        raise ValueError("gdf_nei must include 'district_id' and preferably 'district' to dissolve")
    g = ensure_wgs84(gdf_nei)
    dissolved = g.dissolve(by=cols, as_index=False)
    dissolved = compute_area_km2(dissolved)
    return dissolved
