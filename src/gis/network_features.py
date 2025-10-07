# src/gis/network_features.py
import os
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx

def load_graph(place: str = "Barcelona, Spain", network_type: str = "drive"):
    """
    Download or load the street network for a given place from OpenStreetMap.

    Args:
        place: Place name or query understood by OSMnx (e.g., "Barcelona, Spain").
        network_type: Type of network to download (e.g., "drive", "walk", "bike").

    Returns:
        A NetworkX MultiDiGraph representing the street network.
    """
    ox.settings.use_cache = True  # cache downloads to speed up repeated runs
    G = ox.graph_from_place(place, network_type=network_type)
    return G

def compute_betweenness(G, k: int = 200, weight: str = "length"):
    """
    Compute node betweenness centrality on the graph.

    Args:
        G: NetworkX graph (e.g., MultiDiGraph from OSMnx).
        k: Number of sample nodes for the approximation (speeds up computation).
        weight: Edge attribute to use as weight (e.g., "length").

    Returns:
        A dict {node_id: centrality_value}.
    """
    C = nx.betweenness_centrality(G, k=k, normalized=True, weight=weight)
    return C

def attach_centrality_to_sensors(sensors_gdf: gpd.GeoDataFrame, G, centrality: dict) -> gpd.GeoDataFrame:
    """
    Attach to each sensor the betweenness centrality of its nearest graph node.

    Args:
        sensors_gdf: GeoDataFrame with Point geometries in any CRS.
        G: NetworkX graph (projected or in WGS84; nearest_nodes expects lon/lat).
        centrality: Dict mapping node_id -> centrality value.

    Returns:
        GeoDataFrame with two new columns:
            - nearest_node: id of the nearest graph node.
            - street_centrality: centrality value of that node (0.0 if missing).
    """
    sensors_gdf = sensors_gdf.to_crs(4326)
    nodes = ox.distance.nearest_nodes(
        G,
        sensors_gdf.geometry.x.values,
        sensors_gdf.geometry.y.values
    )
    sensors_gdf["nearest_node"] = nodes
    sensors_gdf["street_centrality"] = sensors_gdf["nearest_node"].map(centrality).fillna(0.0)
    return sensors_gdf


def export_edges_with_centrality(G, centrality: dict, out_path: str = "data/gis_layers/barcelona_centrality.gpkg"):
    """
    Export graph edges with an average centrality derived from their incident nodes.

    The edge centrality is computed as the mean of the centrality of its endpoints (u and v).

    Args:
        G: NetworkX graph.
        centrality: Dict {node_id: centrality_value}.
        out_path: Output path for the GIS file (GeoPackage by default).

    Returns:
        GeoDataFrame of edges with a 'centrality_edge' column.
    """
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_edges["centrality_edge"] = (
        gdf_edges["u"].map(centrality).fillna(0.0) +
        gdf_edges["v"].map(centrality).fillna(0.0)
    ) / 2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf_edges.to_file(out_path)
    return gdf_edges