# scripts/upload_noise_measures.py
import os
import math
import time
import pandas as pd
from tqdm import tqdm
from supabase import create_client, Client
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
TABLE_NAME = "noise"
ROOT_DIR = "data/raw/noise_measures"

# --- Tunables ---
BATCH_SIZE = 1000          # Adjust between 500–2000 depending on network speed
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2  # Seconds between retries


def get_csv_files(root_dir: str) -> List[str]:
    """
    Recursively walk through all subfolders and collect every CSV file.
    """
    csv_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, f))
    return csv_paths


def chunk(records: List[Dict], size: int):
    """
    Yield chunks of records with a specific batch size.
    """
    for i in range(0, len(records), size):
        yield records[i : i + size]


def insert_batch(sb: Client, rows: List[Dict]):
    """
    Insert a batch of rows into the Supabase table.
    """
    return sb.table(TABLE_NAME).insert(rows, returning="minimal").execute()


def main():
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    csv_paths = get_csv_files(ROOT_DIR)
    print(f"📂 {len(csv_paths)} CSV files detected under {ROOT_DIR}")

    for path in csv_paths:
        rel_name = os.path.relpath(path, ROOT_DIR)
        try:
            df = pd.read_csv(path)

            # (Optional) normalize column names if necessary
            # df = df.rename(columns={
            #     "Any": "year", "Mes": "month", "Dia": "day", "Hora": "hour",
            #     "Id_Instal": "id_instal", "Nivell_LAeq_1min": "noise_level_laeq_1min"
            # })

            if df.empty:
                print(f"⚠️  Empty file skipped: {rel_name}")
                continue

            records = df.to_dict(orient="records")
            total = len(records)
            n_batches = math.ceil(total / BATCH_SIZE)

            with tqdm(total=n_batches, desc=f"Inserting {rel_name}") as pbar:
                for rows in chunk(records, BATCH_SIZE):
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            insert_batch(sb, rows)
                            break
                        except Exception as e:
                            msg = str(e)
                            # Retry only for timeouts or transient errors
                            if "57014" in msg or "timeout" in msg.lower():
                                if attempt < MAX_RETRIES:
                                    time.sleep(SLEEP_BETWEEN_RETRIES)
                                    continue
                            # Raise the error if it's not recoverable
                            raise
                    pbar.update(1)

            print(f"✅ Uploaded: {rel_name} ({total} rows)")
        except Exception as e:
            print(f"❌ Error in {rel_name}: {e}")


if __name__ == "__main__":
    main()
