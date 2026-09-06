
#  Download and unzip the penn fudan dataset.
# !wget -q https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

import os
import zipfile
zip_file_path = f"{os.getcwd()}/PennFudanPed.zip"
target_directory = f"{os.getcwd()}/data"
os.makedirs(target_directory, exist_ok=True)

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(target_directory)
