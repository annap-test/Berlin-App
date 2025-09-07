import geopandas as gpd
from shapely.geometry import Polygon

from berlin_labels.geo import compute_area_km2, ensure_wgs84


def test_compute_area_km2_square():
    # Roughly 1km x 1km square near Berlin in EPSG:4326 (approx.)
    # We'll create a small square in meters using a projected CRS then project back
    poly = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    gdf = gpd.GeoDataFrame({"district_id": [1], "neighborhood_id": [1], "neighborhood": ["Test"]}, geometry=[poly], crs=25833)
    gdf = ensure_wgs84(gdf)  # now WGS84
    gdf = compute_area_km2(gdf)
    assert "area_km2" in gdf.columns
    assert 0.95 <= float(gdf.loc[0, "area_km2"]) <= 1.05

