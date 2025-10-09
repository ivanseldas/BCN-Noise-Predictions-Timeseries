# src/gis/spatial_features.py
"""
Spatial features: compute distances from noise sensors to parks and main roads.
"""

import geopandas as gpd
import osmnx as ox
from shapely.ops import unary_union


def load_osm_layers(place="Barcelona, Spain"):
    """Download parks and road network layers from OpenStreetMap."""
    ox.settings.use_cache = True
    tags_green = {"leisure": ["park", "garden"], "landuse": ["grass", "forest"]}
    gdf_green = ox.geometries_from_place(place, tags_green)
    G = ox.graph_from_place(place, network_type="drive")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    return gdf_green, gdf_edges


def add_distance_features(sensors_gdf, gdf_green, gdf_edges):
    """Compute distance to nearest park and main road."""
    sensors_proj = sensors_gdf.to_crs(3857)
    parks_proj = gdf_green.to_crs(3857)
    roads_proj = gdf_edges.to_crs(3857)

    # --- Distance to parks
    parks_union = unary_union(parks_proj.geometry)
    sensors_proj["dist_to_park_m"] = sensors_proj.geometry.distance(parks_union)

    # --- Distance to main roads
    major = roads_proj[roads_proj["highway"].astype(str).str.contains("motorway|primary|trunk", na=False)]
    major_union = unary_union(major.geometry)
    sensors_proj["dist_to_mainroad_m"] = sensors_proj.geometry.distance(major_union)

    return sensors_proj.to_crs(4326)
