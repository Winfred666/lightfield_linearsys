import os
import re
import shutil
from pathlib import Path

def rename_files():
    data_dir = Path("data/raw/lightsheet_vol_6.9")
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    files = list(data_dir.glob("Interp_Vol_ID_*.pt"))
    if not files:
        print("No .pt files found.")
        return

    # Sort files by the integer index in the filename
    def get_index(filepath):
        match = re.search(r"Interp_Vol_ID_(\d+)\.pt", filepath.name)
        if match:
            return int(match.group(1))
        return -1

    files.sort(key=get_index)

    print(f"Found {len(files)} files.")
    if not files:
        return

    first_index = get_index(files[0])
    print(f"First index: {first_index}")
    
    # We want to start from 1.
    # We need to be careful not to overwrite existing files if the new range overlaps with the old range.
    # Range 140-260 maps to 1-121. No overlap.
    
    offset = first_index - 1
    print(f"Offset to subtract: {offset}")

    for file_path in files:
        old_index = get_index(file_path)
        new_index = old_index - offset
        new_name = f"Interp_Vol_ID_{new_index}.pt"
        new_path = data_dir / new_name
        
        print(f"Renaming {file_path.name} -> {new_name}")
        file_path.rename(new_path)

if __name__ == "__main__":
    rename_files()
