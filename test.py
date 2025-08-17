import os
import shutil

source_dir = r"C:\Users\tam\Documents\data\alo_key_1_30"

for filename in os.listdir(source_dir):
    if not filename.lower().endswith(('.jpg')):
        continue

    parts = filename.split("_")
    if len(parts) < 3:
        continue

    L_part = parts[1]
    V_part = parts[2]

    parts[0] = L_part
    new_filename = "_".join(parts)

    v_folder_name = V_part
    v_folder_path = os.path.join(source_dir, v_folder_name)
    os.makedirs(v_folder_path, exist_ok=True)

    old_path = os.path.join(source_dir, filename)
    new_path = os.path.join(v_folder_path, new_filename)
    shutil.copy(old_path, new_path)

    print(f"Moved: {filename} → {new_path}")
