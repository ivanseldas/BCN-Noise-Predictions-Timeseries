"""
arcgis_integration.py
---------------------------------
Utilities to integrate BCN-Noise-Project outputs with ArcGIS ecosystem.
Supports:
    - Exporting CSV/Parquet predictions to Geodatabase (via ArcPy)
    - Uploading predictions to ArcGIS Online as Hosted Feature Layer
    - Updating existing Feature Services with new data
"""

import os
import pandas as pd
import arcpy
from arcgis.gis import GIS
from arcgis.features import FeatureLayer, FeatureSet
from arcgis.geometry import Geometry

# -----------------------------------------------------------
# 1. EXPORT TO GEODATABASE (.GDB)
# -----------------------------------------------------------

def export_to_gdb(input_csv: str, output_gdb: str, layer_name: str = "noise_predictions") -> str:
    """
    Converts a CSV file with Lat/Lon columns into a Feature Class
    stored inside a Geodatabase (.gdb).
    
    Parameters
    ----------
    input_csv : str
        Path to CSV containing columns ['Latitud', 'Longitud'] or ['latitude', 'longitude'].
    output_gdb : str
        Path to output geodatabase.
    layer_name : str
        Name of the output feature class.
        
    Returns
    -------
    output_fc : str
        Full path to the created feature class.
    """
    print(f"📦 Exporting {input_csv} → {output_gdb}/{layer_name}")
    arcpy.env.overwriteOutput = True

    spatial_ref = arcpy.SpatialReference(4326)  # WGS 84
    x_field = "Longitud" if "Longitud" in pd.read_csv(input_csv).columns else "longitude"
    y_field = "Latitud" if "Latitud" in pd.read_csv(input_csv).columns else "latitude"

    arcpy.management.MakeXYEventLayer(input_csv, x_field, y_field, "temp_layer", spatial_ref)
    output_fc = os.path.join(output_gdb, layer_name)
    arcpy.management.CopyFeatures("temp_layer", output_fc)

    print(f"✅ Feature class created: {output_fc}")
    return output_fc


# -----------------------------------------------------------
# 2. UPLOAD TO ARCGIS ONLINE (CREATE NEW LAYER)
# -----------------------------------------------------------

def upload_to_arcgis_online(csv_path: str, gis_user: str, gis_pass: str, title: str, tags: str = "Noise, Prediction, Barcelona") -> str:
    """
    Uploads a CSV (with lat/lon) as a Hosted Feature Layer to ArcGIS Online.
    
    Parameters
    ----------
    csv_path : str
        Path to local CSV file.
    gis_user : str
        ArcGIS Online username.
    gis_pass : str
        ArcGIS Online password.
    title : str
        Title for the hosted layer.
    tags : str
        Metadata tags.
        
    Returns
    -------
    layer_url : str
        URL of the hosted feature layer.
    """
    print("🌐 Connecting to ArcGIS Online...")
    gis = GIS("https://www.arcgis.com", gis_user, gis_pass)

    print(f"⬆️ Uploading {csv_path} as hosted feature layer...")
    item_properties = {
        "title": title,
        "tags": tags,
        "type": "CSV",
    }

    csv_item = gis.content.add(item_properties=item_properties, data=csv_path)
    published_item = csv_item.publish()

    print(f"✅ Published Feature Layer: {published_item.title}")
    print(f"🔗 URL: {published_item.layers[0].url}")

    return published_item.layers[0].url


# -----------------------------------------------------------
# 3. UPDATE EXISTING FEATURE SERVICE
# -----------------------------------------------------------

def update_existing_feature_service(csv_path: str, feature_service_url: str, gis_user: str, gis_pass: str):
    """
    Updates an existing ArcGIS Feature Service with new prediction data.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV with Lat/Lon and predicted values.
    feature_service_url : str
        URL of existing Feature Service layer.
    gis_user : str
        ArcGIS Online username.
    gis_pass : str
        ArcGIS Online password.
    """
    gis = GIS("https://www.arcgis.com", gis_user, gis_pass)
    layer = FeatureLayer(feature_service_url)
    df = pd.read_csv(csv_path)

    print(f"🧭 Preparing {len(df)} new features...")
    features = []
    for _, row in df.iterrows():
        geom = Geometry({
            "x": float(row["Longitud"]),
            "y": float(row["Latitud"]),
            "spatialReference": {"wkid": 4326}
        })
        feature = {"geometry": geom, "attributes": row.to_dict()}
        features.append(feature)

    print("🚀 Uploading new features to existing Feature Service...")
    layer.edit_features(adds=features)
    print("✅ Feature Service successfully updated.")


# -----------------------------------------------------------
# 4. EXAMPLE USAGE (for manual test)
# -----------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()   

GIS_USER = os.getenv("GIS_USER")
GIS_PASS = os.getenv("GIS_PASS")

if __name__ == "__main__":
    # Example configuration (replace with your own)
    
    INPUT_CSV = r"data/processed/noise_predictions.csv"
    OUTPUT_GDB = r"data/gdb/noise_project.gdb"
    GDB_LAYER = "noise_predictions"
    GIS_USER = "your_username"
    GIS_PASS = "your_password"
    TITLE = "Barcelona Noise Forecast 2025"

    # 1. Export to GDB
    export_to_gdb(INPUT_CSV, OUTPUT_GDB, GDB_LAYER)

    # 2. Upload to ArcGIS Online
    upload_to_arcgis_online(INPUT_CSV, GIS_USER, GIS_PASS, TITLE)
