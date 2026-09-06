import shutil
import kagglehub
from pathlib import Path

# Download the PlantVillage dataset
download_path = Path(kagglehub.dataset_download("emware/plantvillage-disease-classification-dataset"))

print(f"Downloaded files to: {download_path}")
