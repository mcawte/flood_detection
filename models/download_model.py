# models/download_model.py
from huggingface_hub import hf_hub_download
import os

MODEL_REPO = "ibm-granite/granite-geospatial-uki-flooddetection"
CKPT_FILE = "granite_geospatial_uki_flood_detection_v1.ckpt"
CONFIG_FILE = "config.yaml"
TARGET_DIR = "." # Download to the directory where the script is run

print(f"Downloading {CKPT_FILE}...")
hf_hub_download(
    repo_id=MODEL_REPO,
    filename=CKPT_FILE,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False
)
print("Checkpoint download complete.")

print(f"Downloading {CONFIG_FILE}...")
hf_hub_download(
    repo_id=MODEL_REPO,
    filename=CONFIG_FILE,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False
)
print("Config download complete.")