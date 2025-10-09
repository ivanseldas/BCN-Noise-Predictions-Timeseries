# src/gis/network_features.py
"""
Network features: compute street betweenness centrality and attach to sensors.
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd


def load_graph(place="Barcelona, Spain", network_type="drive"):
    """Load or download the street network for a given city."""
    ox.settings.use_cache = True
    return ox.graph_from_place(place, network_type=network_type)


def compute_betweenness(G, k=200, weight="length"):
    """Approximate node betweenness centrality."""
    print("⚙️ Computing betweenness centrality ...")
    return nx.betweenness_centrality(G, k=k, normalized=True, weight=weight)


def attach_centrality(sensors_gdf: gpd.GeoDataFrame, G, centrality: dict):
    """Attach nearest node centrality to each sensor."""
    sensors_gdf = sensors_gdf.to_crs(4326)
    nodes = ox.distance.nearest_nodes(
        G,
        sensors_gdf.geometry.x,
        sensors_gdf.geometry.y
    )
    sensors_gdf["nearest_node"] = nodes
    sensors_gdf["street_centrality"] = sensors_gdf["nearest_node"].map(centrality).fillna(0.0)
    return sensors_gdf
