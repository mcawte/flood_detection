import os
import rasterio
from osgeo import gdal

# Print GDAL version
print(f"GDAL version: {gdal.__version__}")

# Check directory structure
print("\nChecking directory structure:")
input_dir = "/app/data/input"
print(f"Input directory exists: {os.path.exists(input_dir)}")

# List files in input directory
print("\nFiles in input directory:")
if os.path.exists(input_dir):
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            print(f"  Dir: {item}")
            # List contents of subdirectories
            for subitem in os.listdir(item_path):
                print(f"    {subitem}")
        else:
            print(f"  File: {item}")

# Try to open a sample TIF file if exists
print("\nTrying to open a sample TIF file:")
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".tif"):
            try:
                sample_path = os.path.join(root, file)
                print(f"Attempting to open: {sample_path}")
                with rasterio.open(sample_path) as src:
                    print(f"  Success! Dimensions: {src.width}x{src.height}")
                    print(f"  Bands: {src.count}")
                    print(f"  CRS: {src.crs}")
                break
            except Exception as e:
                print(f"  Error opening file: {e}")