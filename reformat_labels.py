import json
import os

labels_dir = r"c:\Users\Asus\Downloads\CholecT50-Challenge-Validation\cholect50-challenge-val\labels"
files_to_format = ["VID68.json", "VID70.json", "VID73.json"]

for filename in files_to_format:
    file_path = os.path.join(labels_dir, filename)
    if os.path.exists(file_path):
        print(f"Formatting {filename}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Done formatting {filename}.")
    else:
        print(f"File {filename} not found.")
