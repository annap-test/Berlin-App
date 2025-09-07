from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import folium
import branca


CAT_PALETTE = {
    # High/positive
    "well-connected": "#1a9641",  # deep green
    "above average": "#1a9641",
    "vibrant": "#1a9641",
    # Mid
    "moderate": "#a6d96a",       # light green
    "average": "#a6d96a",
    # Low/negative — use yellow instead of gray/red
    "remote": "#fee08b",
    "below average": "#fee08b",
    "sparse": "#fee08b",
}


def _make_base_map() -> folium.Map:
    return folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="cartodbpositron")


def _tooltip_fields_and_aliases(gdf: gpd.GeoDataFrame, extras: Iterable[str]) -> Tuple[List[str], List[str]]:
    fields: List[str] = []
    aliases: List[str] = []
    if "neighborhood" in gdf.columns:
        fields.append("neighborhood")
        aliases.append("Neighborhood")
    elif "neighborhood_id" in gdf.columns:
        fields.append("neighborhood_id")
        aliases.append("Neighborhood ID")
    if "district" in gdf.columns:
        fields.append("district")
        aliases.append("District")
    for c in extras:
        if c in gdf.columns and c not in fields:
            fields.append(c)
            # Pretty alias from column name
            aliases.append(c.replace("_", " ").title())
    return fields, aliases


def make_categorical_map(gdf: gpd.GeoDataFrame, category_columns: List[str]) -> folium.Map:
    m = _make_base_map()
    for col in category_columns:
        if col not in gdf.columns:
            continue
        def style_fn(feature, column=col):
            v = feature["properties"].get(column)
            color = CAT_PALETTE.get(str(v).lower(), "#cccccc")
            return {"fillColor": color, "color": "#555555", "weight": 0.5, "fillOpacity": 0.7}

        layer = folium.FeatureGroup(name=f"{col}", show=False)
        fields, aliases = _tooltip_fields_and_aliases(gdf, [col])
        folium.GeoJson(
            gdf,
            name=col,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, labels=True),
            highlight_function=lambda feat: {"weight": 2, "color": "#333333"},
        ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _numeric_colormap(lo: float, hi: float):
    # Custom gradient (low -> high): gray/light yellow -> light green -> deep green
    colors = [
        "#bdbdbd",  # neutral gray for very low
        "#ffffe5",  # near white
        "#fff7bc",  # pale yellow
        "#d9f0a3",  # light green
        "#66bd63",  # medium green
        "#1a9641",  # deep green for high
    ]
    return branca.colormap.LinearColormap(colors=colors, vmin=lo, vmax=hi)


def make_numeric_map(gdf: gpd.GeoDataFrame, numeric_columns: List[str]) -> folium.Map:
    m = _make_base_map()
    for col in numeric_columns:
        if col not in gdf.columns:
            continue
        s = gdf[col].astype(float)
        if s.notna().sum() == 0:
            continue
        lo = float(np.nanpercentile(s, 5))
        hi = float(np.nanpercentile(s, 95))
        cmap = _numeric_colormap(lo, hi).to_step(11)

        def style_fn(feature, column=col, color_map=cmap):
            v = feature["properties"].get(column)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = None
            color = color_map(v) if v is not None else "#cccccc"
            return {"fillColor": color, "color": "#555555", "weight": 0.5, "fillOpacity": 0.75}

        layer = folium.FeatureGroup(name=f"{col}", show=False)
        fields, aliases = _tooltip_fields_and_aliases(gdf, [col])
        folium.GeoJson(
            gdf,
            name=col,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, labels=True),
            highlight_function=lambda feat: {"weight": 2, "color": "#333333"},
        ).add_to(layer)
        layer.add_to(m)
        cmap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def make_quickcheck_maps(gdf: gpd.GeoDataFrame) -> Dict[str, folium.Map]:
    cat_cols = [c for c in ["mobility_label", "green_share_label", "playgrounds_density_label", "vibrancy_label"] if c in gdf.columns]
    num_cols = [
        c for c in [
            "connectivity_density",
            "green_share",
            "playgrounds_per_km2",
            "venues_per_km2",
            "n_cuisine_types",
            "V_score",
            "C_score",
            "VV_index",
        ] if c in gdf.columns
    ]
    maps = {}
    if cat_cols:
        maps["categorical"] = make_categorical_map(gdf, cat_cols)
    if num_cols:
        maps["numeric"] = make_numeric_map(gdf, num_cols)
    return maps


def save_maps(maps: Dict[str, folium.Map], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in maps.items():
        m.save(str(out / f"quickcheck_{name}.html"))


def make_district_maps(gdf_nei: gpd.GeoDataFrame, dist_gdf: gpd.GeoDataFrame) -> Dict[str, folium.Map]:
    """Build categorical/numeric maps for districts (using same palettes)."""
    maps: Dict[str, folium.Map] = {}
    # Try same columns as neighborhoods but on district polygons
    cat_cols = [c for c in ["mobility_label", "green_share_label", "playgrounds_density_label", "vibrancy_label"] if c in dist_gdf.columns]
    num_cols = [c for c in ["connectivity_density", "green_share", "playgrounds_per_km2", "venues_per_km2", "n_cuisine_types", "V_score", "C_score", "VV_index"] if c in dist_gdf.columns]
    if cat_cols:
        maps["categorical_districts"] = make_categorical_map(dist_gdf, cat_cols)
    if num_cols:
        maps["numeric_districts"] = make_numeric_map(dist_gdf, num_cols)
    return maps


def make_combined_map(nei_gdf: gpd.GeoDataFrame, dist_gdf: gpd.GeoDataFrame, column: str) -> folium.Map:
    """Overlay a single metric/label at both district and neighborhood levels."""
    m = _make_base_map()
    # District layer on bottom
    if column in dist_gdf.columns:
        s = dist_gdf[column]
        if np.issubdtype(s.dtype, np.number):
            lo = float(np.nanpercentile(s, 5))
            hi = float(np.nanpercentile(s, 95))
            cmap = _numeric_colormap(lo, hi).to_step(11)
            def style_d(feature, column=column, color_map=cmap):
                v = feature["properties"].get(column)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = None
                color = color_map(v) if v is not None else "#cccccc"
                return {"fillColor": color, "color": "#333333", "weight": 1.0, "fillOpacity": 0.5}
        else:
            def style_d(feature, column=column):
                v = feature["properties"].get(column)
                color = CAT_PALETTE.get(str(v).lower(), "#cccccc")
                return {"fillColor": color, "color": "#333333", "weight": 1.0, "fillOpacity": 0.5}
        folium.GeoJson(dist_gdf, name=f"districts_{column}", style_function=style_d).add_to(m)
    # Neighborhood overlay
    if column in nei_gdf.columns:
        s = nei_gdf[column]
        if np.issubdtype(s.dtype, np.number):
            lo = float(np.nanpercentile(s, 5))
            hi = float(np.nanpercentile(s, 95))
            cmap = _numeric_colormap(lo, hi).to_step(11)
            def style_n(feature, column=column, color_map=cmap):
                v = feature["properties"].get(column)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = None
                color = color_map(v) if v is not None else "#cccccc"
                return {"fillColor": color, "color": "#111111", "weight": 0.5, "fillOpacity": 0.7}
        else:
            def style_n(feature, column=column):
                v = feature["properties"].get(column)
                color = CAT_PALETTE.get(str(v).lower(), "#cccccc")
                return {"fillColor": color, "color": "#111111", "weight": 0.5, "fillOpacity": 0.7}
        folium.GeoJson(nei_gdf, name=f"neighborhoods_{column}", style_function=style_n).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
