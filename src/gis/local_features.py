# src/gis/local_features.py
"""
Local spatial features: compute local mean of noise within a given radius.
"""

import geopandas as gpd


def add_local_mean(sensors_gdf: gpd.GeoDataFrame, radius_m: float = 150.0, value_col: str = "Avg_Noise_dB"):
    """Compute local mean of noise values within a given radius (m)."""
    sensors_proj = sensors_gdf.to_crs(3857)
    tree = sensors_proj.sindex
    means = []

    for geom in sensors_proj.geometry:
        buffer = geom.buffer(radius_m)
        idx = list(tree.query(buffer, predicate="intersects"))
        means.append(sensors_proj.iloc[idx][value_col].mean())

    sensors_gdf[f"{value_col}_localmean_{radius_m}m"] = means
    return sensors_gdf
