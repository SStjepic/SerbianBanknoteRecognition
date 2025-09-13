import os
from pathlib import Path

LABELS_DIR = Path("data/raw/labels")

for txt_file in LABELS_DIR.glob("*.txt"):

    old_name = txt_file.name
    
    if "-" in old_name:
        new_name = old_name.split("-", 1)[1] 
        
        if new_name.endswith(".jpg.txt"):
            new_name = new_name.replace(".jpg.txt", ".txt")

        new_path = txt_file.parent / new_name
        
        txt_file.rename(new_path)
        print(f"Renamed: {old_name} -> {new_name}")
    else:
        print(f"Skipped (no dash): {old_name}")

print("✅ Successfully renamed all files!")
