import os
import zipfile
import glob

zip_file_dir = "zips"
output_dir = "rpi_data_raw"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

zip_files = os.listdir(zip_file_dir)
for zip_file in zip_files:
    print(f"Unzipping: {zip_file}")
    zip_file_path = os.path.join(zip_file_dir, zip_file)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)