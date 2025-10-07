from src.gis.network_features import compute_network_centrality
from src.gis.spatial_features import add_spatial_context

def build_features(df):
    df = basic_time_features(df)
    df = compute_network_centrality(df)
    df = add_spatial_context(df)
    return df
