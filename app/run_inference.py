import subprocess
import argparse
import sys
import rioxarray
import numpy as np

from glob import glob

files = sorted(glob("/app/data/input/*.tif"))
print(f"Found {len(files)} .tif files")
for f in files:
    try:
        arr = rioxarray.open_rasterio(f, masked=True)
        print(
            f"{f}: shape={arr.shape}, dtype={arr.dtype}, "
            f"NaNs={np.isnan(arr).sum().item()}"
        )
    except Exception as e:
        print(f"Error loading {f}: {e}")
 
print("Starting inference script inside container...")
input_tif = "/app/data/input/EMSR407_AOI_3_2019-11-14_tile_0_2_test_image.tif"
print(rioxarray.open_rasterio(input_tif).shape)
arr = rioxarray.open_rasterio(input_tif, masked=True)
print(np.isnan(arr).sum().item())
 
# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Run Terratorch inference inside Docker."
)
parser.add_argument(
    '--config',
    default='/app/configs/'
            'config_granite_geospatial_uki_flood_detection_v1.yaml',
    help='Path inside container to config.yaml'
)
parser.add_argument(
    '--checkpoint',
    default='/app/models/granite_geospatial_uki_flood_detection_v1.ckpt',
    help='Path inside container to model.ckpt'
)
parser.add_argument(
    '--input_dir',
    default='/app/data/input',
    help='Path inside container to the input data root directory '
         '(parent of image files)'
)
parser.add_argument(
    '--output_dir',
    default='/app/data/output',
    help='Path inside container for prediction output'
)
# Add accelerator argument
parser.add_argument(
    '--accelerator', default='cpu',
    help='Accelerator to use (e.g., cpu, gpu)'
)
args = parser.parse_args()

print(f"Config Path: {args.config}")
print(f"Checkpoint Path: {args.checkpoint}")
print(f"Input Data Root: {args.input_dir}")
print(f"Output Directory: {args.output_dir}")
print(f"Accelerator: {args.accelerator}")

# --- Terratorch Command Construction ---
# We need to run terratorch from the directory containing 'custom_modules'
# Assuming the project code is copied to /app/uki-flooddetection
project_code_dir = "/app"
predict_script = "terratorch"  # Assuming terratorch is in the PATH

command = [
    predict_script,
    "predict",
    "-c", args.config,
    "--ckpt_path", args.checkpoint,
    "--predict_output_dir", args.output_dir,
    "--data.init_args.predict_data_root", args.input_dir,
    "--data.init_args.img_grep", "*.tif",
    f"--trainer.accelerator={args.accelerator}",  # Control CPU/GPU
    "--trainer.devices=1",  # Specify number of devices (1 for CPU/single GPU)
    "--data.init_args.batch_size=1"  # Override batch size for prediction
]

print(f"\nExecuting command: {' '.join(command)}\n")

# --- Execute Command ---
try:
    # Run the command from the project directory
    process = subprocess.Popen(
        command,
        cwd=project_code_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Stream output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    if rc == 0:
        print("\nTerratorch predict command finished successfully.")
    else:
        print(
            f"\nTerratorch predict command failed with exit code {rc}.",
            file=sys.stderr
        )
        sys.exit(rc)  # Exit script with the same error code

except FileNotFoundError:
    print(
        f"Error: Command '{predict_script}' not found. "
        "Is terratorch installed and in PATH?",
        file=sys.stderr
    )
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
    sys.exit(1)

print("\nInference script finished.")
