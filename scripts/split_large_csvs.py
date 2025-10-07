import os
import pandas as pd
from pathlib import Path
import argparse


def split_csv(input_path, output_dir, max_mb=50):
    """
    Split a CSV into multiple parts, each smaller than `max_mb` megabytes.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_size = os.path.getsize(input_path) / (1024 * 1024)

    if file_size <= max_mb:
        print(f"✅ {Path(input_path).name} is {file_size:.1f} MB — no need to split.")
        return [input_path]

    print(f"⚙️ Splitting {Path(input_path).name} ({file_size:.1f} MB) into ~{max_mb} MB chunks...")

    chunksize = 800_000
    base_name = Path(input_path).stem
    output_files = []
    part = 1
    current_size = 0
    rows = []

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        rows.append(chunk)
        current_size += chunk.memory_usage(deep=True).sum() / (1024 * 1024)

        if current_size >= max_mb:
            df = pd.concat(rows)
            out_path = Path(output_dir) / f"{base_name}_part_{part}.csv"
            df.to_csv(out_path, index=False)
            output_files.append(str(out_path))
            print(f"🪓 Saved: {out_path.name} (~{current_size:.1f} MB)")
            part += 1
            rows, current_size = [], 0

    # Last chunk
    if rows:
        df = pd.concat(rows)
        out_path = Path(output_dir) / f"{base_name}_part_{part}.csv"
        df.to_csv(out_path, index=False)
        output_files.append(str(out_path))
        print(f"🪓 Saved: {out_path.name} (last chunk)")

    print(f"✅ {Path(input_path).name} split into {len(output_files)} parts.")
    return output_files


def process_folder(folder_path, max_mb=50):
    """
    Walk through a folder and automatically split CSVs that exceed the given size.
    """
    folder_path = Path(folder_path)
    csv_files = list(folder_path.rglob("*.csv"))

    print(f"📂 Scanning {len(csv_files)} CSV files in '{folder_path}'...\n")

    for csv_file in csv_files:
        file_size = os.path.getsize(csv_file) / (1024 * 1024)
        if file_size > max_mb:
            output_dir = csv_file.parent / "split"
            print(f"🔍 Large file detected: {csv_file.name} ({file_size:.1f} MB)")
            split_csv(csv_file, output_dir, max_mb=max_mb)
        else:
            print(f"✅ {csv_file.name} ({file_size:.1f} MB) does not require splitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large CSV files (>50 MB) in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the folder with CSV files.")
    parser.add_argument("--max_mb", type=int, default=50, help="Maximum allowed size per file (MB).")

    args = parser.parse_args()

    input_folder = Path(args.input_folder)

    print(f"📁 Input folder: {input_folder}")
    print(f"📏 Per-file limit: {args.max_mb} MB\n")

    process_folder(input_folder, max_mb=args.max_mb)
