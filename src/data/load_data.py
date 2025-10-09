import os
import glob
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------
# Batch processing — memory-safe version
# ---------------------------------------------------------------------
def process_and_append_parquet(base_folder: str, output_path: str, start: int = 2015, end: int = 2025):
    """
    Process CSV files year by year and append them to a single Parquet file.
    Efficient for large datasets that don't fit in memory.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    first_write = True

    for year in range(start, end + 1):
        year_path = os.path.join(base_folder, str(year))
        if not os.path.exists(year_path):
            print(f"⚠️ Folder not found: {year_path}")
            continue

        csv_files = sorted(glob.glob(os.path.join(year_path, "*.csv")))
        if not csv_files:
            print(f"⚠️ No CSV files in {year_path}")
            continue

        print(f"📂 Processing {len(csv_files)} files from {year}...")

        for file_path in tqdm(csv_files, desc=f"Year {year}"):
            try:
                # Read CSV in chunks (in case some single file is also large)
                for chunk in pd.read_csv(file_path, chunksize=200_000):
                    chunk.columns = [c.strip() for c in chunk.columns]

                    # unify noise column
                    for col in chunk.columns:
                        lower = col.lower()
                        if "nivell" in lower and "laeq" in lower:
                            if col != "Nivell_LAeq_1h":
                                chunk.rename(columns={col: "Nivell_LAeq_1h"}, inplace=True)
                            break

                    chunk["year"] = year

                    # Append parquet chunk safely
                    chunk.to_parquet(
                        output_path,
                        index=False,
                        engine="fastparquet",
                        compression="snappy",
                        append=not first_write,
                    )
                    first_write = False

            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

    print(f"✅ All files processed and saved to {output_path}")


def main():
    BASE_DIR = "data/raw/noise_measures"
    OUTPUT_PATH = "data/interim/noise_measures_merged.parquet"

    process_and_append_parquet(BASE_DIR, OUTPUT_PATH, start=2015, end=2025)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()