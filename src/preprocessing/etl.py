import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# Initialize Spark session
spark = SparkSession.builder.appName("NoiseDataProcessing").getOrCreate()

# 1. Load data
raw_data_path = '../../data/raw/noise_measures/'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through the files and read them into PySpark DataFrames
for file in csv_files:
    file_path = os.path.join(raw_data_path, file)
    df = spark.read.csv(file_path, header=True)  # Use spark.read.csv
    dataframes.append(df)

# Concatenate all DataFrames using union
df = dataframes[0]  # Start with the first DataFrame
for df_unique in dataframes[1:]:  # Loop through the rest and union them
    df = df_unique.union(df)


# 2. Transformations

try:
    # Format the time
    df = df.withColumn(
        "hora_formatted", 
        concat(
            lpad(split(col("hora"), ":")[0], 2, "0"),  # Extract hours and pad with zeros
            lit(":"),
            split(col("hora"), ":")[1],  # Extract minutes
            lit(":00")  # Add seconds
        )
    )
    
    # Create a timestamp by combining date and formatted time columns
    df = df.withColumn(
        "timestamp",
        to_timestamp(
            concat_ws(" ",  # Combine date and time with a space
                concat_ws("-",  # Combine date parts with hyphens
                    col("any").cast("string"),  # Year
                    lpad(col("mes").cast("string"), 2, "0"),  # Month (padded with zeros)
                    lpad(col("dia").cast("string"), 2, "0")  # Day (padded with zeros)
                ),
                col("hora_formatted")  # Formatted time
            ),
            "yyyy-MM-dd HH:mm:ss"  # Timestamp format
        )
    )
except Exception as e:
    print(f"Error during transformations: {str(e)}")
    raise

# 3. Select final columns
df_final = df.select(
    col("timestamp"),  # Timestamp column
    col("id_instal").alias("sensor_id"),  # Rename 'id_instal' to 'sensor_id'
    col("nivell_laeq_1h").alias("noise_db")  # Rename 'nivell_laeq_1h' to 'noise_db'
).distinct()  # Remove duplicate rows

# 4. Save data locally 
output_path = "../../data/processed/ETL"  # Change the path as needed
try:
    df_final.write.mode("overwrite").partitionBy("sensor_id").parquet(output_path)  # Save as Parquet format
    print(f"Data saved successfully at: {output_path}")
    
except Exception as e:
    print(f"Error saving data: {str(e)}")
    raise

# Stop the Spark session
spark.stop()