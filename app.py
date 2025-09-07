import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from pathlib import Path


st.set_page_config(page_title="Berlin Explorer", layout="wide")


def find_paths():
    root = Path.cwd()
    if not (root / "data").exists():
        root = root.parent
    data_dir = root / "data"
    outputs1 = root / "labels_with_visualization" / "outputs"
    outputs2 = root / "outputs"
    out_dir = outputs1 if outputs1.exists() else outputs2
    return data_dir, out_dir


@st.cache_data(show_spinner=False)
def load_sources():
    data_dir, out_dir = find_paths()
    nei_geo = gpd.read_file(data_dir / "neighborhoods.geojson")
    if nei_geo.crs is None or (nei_geo.crs and nei_geo.crs.to_epsg() != 4326):
        try:
            nei_geo = nei_geo.to_crs(4326)
        except Exception:
            nei_geo = nei_geo.set_crs(4326)
    # Wide tables
    nei_wide = pd.read_csv(out_dir / "berlin_neighborhoods_labels_wide.csv") if (out_dir / "berlin_neighborhoods_labels_wide.csv").exists() else pd.DataFrame()
    dist_wide = pd.read_csv(out_dir / "berlin_districts_labels_wide.csv") if (out_dir / "berlin_districts_labels_wide.csv").exists() else pd.DataFrame()
    return nei_geo, nei_wide, dist_wide, out_dir


def percentile_score(series: pd.Series, lo: int = 10, hi: int = 90) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(s).any():
        return pd.Series(np.nan, index=series.index)
    p_lo = np.nanpercentile(s, lo)
    p_hi = np.nanpercentile(s, hi)
    rng = max(p_hi - p_lo, 1e-9)
    def _s(x):
        if pd.isna(x):
            return np.nan
        return float(np.clip((x - p_lo) / rng, 0, 1)) * 100.0
    return s.apply(_s)


def render_map(gdf: gpd.GeoDataFrame, value_col: str):
    s = pd.to_numeric(gdf[value_col], errors="coerce")
    if s.notna().any():
        p5 = float(np.nanpercentile(s, 5))
        p95 = float(np.nanpercentile(s, 95))
        if p95 <= p5: p95 = p5 + 1e-9
    else:
        p5, p95 = 0.0, 1.0
    cmap = LinearColormap(["#ffffcc", "#a6d96a", "#1a9641"], vmin=p5, vmax=p95)
    cmap.caption = f"{value_col}: scaled p5→p95"

    m = folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="cartodbpositron")
    gj = folium.GeoJson(
        data=gdf.to_json(),
        style_function=lambda f: {
            "fillColor": cmap(float(f["properties"].get(value_col)) if f["properties"].get(value_col) is not None and not pd.isna(f["properties"].get(value_col)) else "#cccccc"),
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=[c for c in ["district", "neighborhood", value_col] if c in gdf.columns], aliases=[c.title() for c in ["district", "neighborhood", value_col] if c in gdf.columns])
    )
    gj.add_to(m)
    cmap.add_to(m)
    return m


def weight_ui(container, items: list[tuple[str, str]], defaults: dict[str, bool], key_prefix: str) -> dict:
    """Render checkboxes with descriptions and per-item sliders below.

    items: list of (label, description)
    returns dict[label] -> weight (0..100)
    """
    weights: dict[str, int] = {}
    for label, desc in items:
        enabled = container.checkbox(label, value=defaults.get(label, False), key=f"{key_prefix}_chk_{label}")
        container.caption(desc)
        if enabled:
            weights[label] = container.slider("Importance", 0, 100, 50, key=f"{key_prefix}_w_{label}")
        else:
            weights[label] = 0
        # compact divider
        container.markdown("<hr style='margin:4px 0; opacity:0.2;'>", unsafe_allow_html=True)
    return weights


st.title("Berlin Explorer – Weighted Suitability")
st.caption("Pick level and features, adjust importance, then press Show to highlight the most suitable areas.")

# Compact UI spacing without changing font sizes
st.markdown(
    """
    <style>
    /* Reduce vertical gaps between widgets */
    div[data-testid="stCheckbox"] { margin-bottom: 0.15rem; }
    div[data-testid="stSlider"] { margin-top: 0.10rem; margin-bottom: 0.35rem; }
    /* Reduce section paddings slightly */
    section.main > div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

nei_geo, nei_wide, dist_wide, out_dir = load_sources()

level = st.radio("Level", ["Neighborhoods", "Districts"], horizontal=True)

def compute_component(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if invert:
        s = -s
    return percentile_score(s)


left_col, right_col = st.columns([1, 2], gap="large")

if level == "Neighborhoods":
    # Merge polygons + metrics
    if not nei_wide.empty:
        g = nei_geo.merge(nei_wide, on=["district", "neighborhood"], how="left")
    else:
        g = nei_geo.copy()
    # Feature set
    feature_map = {
        "Venues vibrancy": "VV_index" if "VV_index" in g.columns else ("venues_per_km2" if "venues_per_km2" in g.columns else None),
        "Mobility": "mobility_score" if "mobility_score" in g.columns else ("connectivity_density" if "connectivity_density" in g.columns else None),
        "Playgrounds density": "playgrounds_per_km2" if "playgrounds_per_km2" in g.columns else None,
        "Green share": "green_share" if "green_share" in g.columns else None,
    }
    features = [k for k, v in feature_map.items() if v is not None]
    st.subheader("Neighborhood features")
    # Descriptions for neighborhood metrics
    nei_desc = {
        "Venues vibrancy": "Combined venue density and variety; higher = more lively.",
        "Mobility": "Public transport accessibility (U‑Bahn + bus/tram); higher = better connected.",
        "Playgrounds density": "Playgrounds per km²; higher = more playground access.",
        "Green share": "Parks + forest as percentage of district area; higher = greener.",
    }
    items = [(name, nei_desc.get(name, "")) for name in features]
    weights_raw = weight_ui(left_col, items, defaults={"Venues vibrancy": True, "Mobility": True}, key_prefix="nei")
    if left_col.button("Show", key="nei_show"):
        comp_parts = []
        for name in features:
            w = float(weights_raw.get(name, 0))
            if w <= 0:
                continue
            col = feature_map[name]
            comp_parts.append(percentile_score(g[col]) * (w / 100.0))
        g_res = g.copy()
        g_res["suitability"] = pd.concat(comp_parts, axis=1).sum(axis=1) if comp_parts else 0.0
        st.session_state["nei_result"] = g_res
    if "nei_result" in st.session_state:
        g_show = st.session_state["nei_result"]
        m = render_map(g_show, "suitability")
        right_col.subheader("Map")
        st_folium(m, height=700, use_container_width=True, returned_objects=[])
        top = g_show[["district", "neighborhood", "suitability"]].sort_values("suitability", ascending=False).head(20)
        right_col.subheader("Top neighborhoods")
        right_col.dataframe(top, use_container_width=True)
else:
    # District polygons from neighborhoods
    dist_geo = nei_geo[["district", "geometry"]].dissolve(by="district", as_index=False)
    g = dist_geo.merge(dist_wide, on="district", how="left") if not dist_wide.empty else dist_geo.copy()
    # Feature set (exclude combined composites like income_safety, urbanity)
    # Add metrics requested by user perspective
    feature_map = {
        "Income": ("income_value_eur", False) if "income_value_eur" in g.columns else None,
        "Safety": ("crimes_per_1000", True) if "crimes_per_1000" in g.columns else None,
        "Unemployment": ("unemployment_per_1000", True) if "unemployment_per_1000" in g.columns else None,
        "Density": ("density_per_km2", False) if "density_per_km2" in g.columns else None,
        "Diversity": ("diversity_share", False) if "diversity_share" in g.columns else None,
        "Green share": ("green_share", False) if "green_share" in g.columns else None,
        "Vibrancy": ("VV_index", False) if "VV_index" in g.columns else (("venues_per_km2", False) if "venues_per_km2" in g.columns else None),
        "Mobility": ("mobility_score", False) if "mobility_score" in g.columns else (("connectivity_density", False) if "connectivity_density" in g.columns else None),
        "Playgrounds density": ("playgrounds_per_km2", False) if "playgrounds_per_km2" in g.columns else None,
    }
    features = [k for k, v in feature_map.items() if v is not None]
    st.subheader("District features")
    dist_desc = {
        "Income": "Median income of residents; higher = more affluent.",
        "Safety": "Crimes per 1,000 residents; lower = safer.",
        "Unemployment": "Unemployed per 1,000 residents; lower = better employment.",
        "Density": "Residents per km²; higher = more urban/compact.",
        "Diversity": "Share of residents with migrant background or non‑German citizenship; higher = more diverse.",
        "Green share": "Parks + forest as percentage of district area; higher = greener.",
        "Vibrancy": "Combined venue density and variety; higher = more lively.",
        "Mobility": "Public transport accessibility (U‑Bahn + bus/tram); higher = better connected.",
        "Playgrounds density": "Playgrounds per km²; higher = more playground access.",
    }
    dist_items = [(name, dist_desc.get(name, "")) for name in features]
    weights_raw = weight_ui(left_col, dist_items, defaults={"Green share": True}, key_prefix="dist")
    if left_col.button("Show", key="dist_show"):
        comp_parts = []
        for name in features:
            w = float(weights_raw.get(name, 0))
            if w <= 0:
                continue
            col, invert = feature_map[name]
            comp_parts.append(compute_component(g[col], invert=invert) * (w / 100.0))
        g_res = g.copy()
        g_res["suitability"] = pd.concat(comp_parts, axis=1).sum(axis=1) if comp_parts else 0.0
        st.session_state["dist_result"] = g_res
    if "dist_result" in st.session_state:
        g_show = st.session_state["dist_result"]
        m = render_map(g_show, "suitability")
        right_col.subheader("Map")
        st_folium(m, height=700, use_container_width=True, returned_objects=[])
        top = g_show[["district", "suitability"]].sort_values("suitability", ascending=False).head(20)
        right_col.subheader("Top districts")
        right_col.dataframe(top, use_container_width=True)

st.markdown("---")
st.caption("Notes: Each selected feature is scaled to 0–100 via percentile scaling (p10→0, p90→100). Weights are linear importances; suitability is the weighted sum.")
