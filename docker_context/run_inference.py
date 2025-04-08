import subprocess
import argparse
import sys

print("Starting inference script inside container...")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Run Terratorch inference inside Docker."
)
parser.add_argument(
    '--config', required=True, help='Path inside container to config.yaml'
)
parser.add_argument(
    '--checkpoint', required=True, help='Path inside container to model.ckpt'
)
parser.add_argument(
    '--input_dir', required=True,
    help='Path inside container to the input data root directory '
         '(parent of image files)'
)
parser.add_argument(
    '--output_dir', required=True,
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
project_code_dir = "/app/uki-flooddetection"
predict_script = "terratorch"  # Assuming terratorch is in the PATH

command = [
    predict_script,
    "predict",
    "-c", args.config,
    "--ckpt_path", args.checkpoint,
    "--predict_output_dir", args.output_dir,
    "--data.init_args.predict_data_root", args.input_dir,
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
