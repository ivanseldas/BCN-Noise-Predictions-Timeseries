# src/gis/spatial_features.py
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points

def sjoin_nearest_landuse(
    sensors_gdf: gpd.GeoDataFrame,
    landuse_gdf: gpd.GeoDataFrame,
    landuse_col: str = "landuse_type"
) -> gpd.GeoDataFrame:
    """
    Attach the nearest land use type to each sensor.

    Args:
        sensors_gdf: GeoDataFrame with sensor points.
        landuse_gdf: GeoDataFrame with land use polygons.
        landuse_col: Column in landuse_gdf containing land use labels.

    Returns:
        GeoDataFrame of sensors with 'landuse_type' and 'dist_landuse' columns.
    """
    sensors_gdf = sensors_gdf.to_crs(landuse_gdf.crs)
    out = gpd.sjoin_nearest(
        sensors_gdf,
        landuse_gdf[[landuse_col, "geometry"]],
        how="left",
        distance_col="dist_landuse"
    )
    return out


def distance_to_primary_roads(
    sensors_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    highway_levels: tuple = ("motorway", "trunk", "primary")
) -> gpd.GeoDataFrame:
    """
    Compute distance from each sensor to the nearest major road.

    Args:
        sensors_gdf: GeoDataFrame of sensors.
        edges_gdf: GeoDataFrame of road edges from OSMnx.
        highway_levels: Road types considered 'major'.

    Returns:
        GeoDataFrame with 'dist_primary_m' column.
    """
    if "highway" not in edges_gdf.columns:
        edges_gdf["highway"] = None

    # Filter only major roads
    prim = edges_gdf[
        edges_gdf["highway"]
        .astype(str)
        .str.contains("|".join(highway_levels), case=False, na=False)
    ]

    sensors_proj = sensors_gdf.to_crs(3857)
    prim_proj = prim.to_crs(3857)

    # Compute distance to the union of major roads
    prim_union = prim_proj.unary_union
    sensors_proj["dist_primary_m"] = sensors_proj.geometry.distance(prim_union)

    return sensors_proj.to_crs(sensors_gdf.crs)


def local_noise_mean(
    sensors_gdf: gpd.GeoDataFrame,
    radius_m: float = 100.0,
    value_col: str = "noise_pred"
) -> gpd.GeoDataFrame:
    """
    Compute local mean of a value within a given radius.

    Args:
        sensors_gdf: GeoDataFrame with sensor points.
        radius_m: Radius (in meters) for neighborhood search.
        value_col: Column whose local mean will be computed.

    Returns:
        GeoDataFrame with new column '{value_col}_local_mean_<radius>m'.
    """
    sensors_proj = sensors_gdf.to_crs(3857)
    tree = sensors_proj.sindex

    vals = []
    for geom in sensors_proj.geometry:
        buffer = geom.buffer(radius_m)
        neighbors_idx = list(tree.query(buffer, predicate="intersects"))
        vals.append(sensors_proj.iloc[neighbors_idx][value_col].mean())

    sensors_gdf[f"{value_col}_local_mean_{int(radius_m)}m"] = vals
    return sensors_gdf
