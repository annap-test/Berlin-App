"""Berlin neighborhood labeling utilities.

Modules:
- text: canonicalization and cuisine tokenization
- geo: CRS helpers, area calculations, spatial joins
- io: load/save helpers for CSV/GeoJSON
- labels_*: themed feature/label computations
"""

from .text import canon_nh, tokenize_cuisines, national_cuisine_vocab
from .geo import ensure_wgs84, compute_area_km2, to_points_gdf, points_within
from .io import load_neighborhoods, load_csv, save_geodataframe, save_dataframe

__all__ = [
    "canon_nh",
    "tokenize_cuisines",
    "national_cuisine_vocab",
    "ensure_wgs84",
    "compute_area_km2",
    "to_points_gdf",
    "points_within",
    "load_neighborhoods",
    "load_csv",
    "save_geodataframe",
    "save_dataframe",
]

