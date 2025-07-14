import gradio as gr
import os
import base64
import tempfile
import requests
# import shutil
import subprocess
from pathlib import Path
import torch
import sys
import boto3

# Define fixed paths from your container setup
CONFIG_PATH = '/app/configs/config_granite_geospatial_uki_flood_detection_v1.yaml'
CHECKPOINT_PATH = '/app/models/granite_geospatial_uki_flood_detection_v1.ckpt'
PROJECT_CODE_DIR = "/app"

# --- MinIO Configuration ---
MINIO_ENDPOINT = 'https://minio-s3-ppe-multi-modal.apps.cluster-r8fxn.r8fxn.sandbox753.opentlc.com'
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
MINIO_BUCKET = 'flood-predictions'


def upload_to_minio(file_path: str, object_name: str) -> str:
    """
    Upload a file to MinIO storage.
    """
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise gr.Error(
            "MinIO credentials are not configured. Cannot upload result.")

    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name='us-east-1'
        )
        print(
            f"Uploading {file_path} to MinIO bucket '{MINIO_BUCKET}' as '{object_name}'")
        s3_client.upload_file(file_path, MINIO_BUCKET, object_name)

        # Generate a presigned URL that expires in 1 hour (3600 seconds)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': MINIO_BUCKET, 'Key': object_name},
            ExpiresIn=3600
        )
        print(f"File uploaded. Shareable URL: {presigned_url}")
        return presigned_url
    except Exception as e:
        print(f"Error uploading to MinIO: {e}", file=sys.stderr)
        raise gr.Error(f"Failed to upload result to MinIO: {e}")
    

def ensure_files_exist():
    """
    On startup, check if the config and model files exist locally.
    If not, download them from the 'flood-models' MinIO bucket.
    """
    files_to_check = {
        CONFIG_PATH: "config_granite_geospatial_uki_flood_detection_v1.yaml",
        CHECKPOINT_PATH: "granite_geospatial_uki_flood_detection_v1.ckpt",
    }

    # Don't try to download if credentials aren't set
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        print("WARNING: MinIO credentials not found. Cannot check or download model files.")
        return

    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
        verify=False  # As determined in previous steps
    )

    for local_path, minio_filename in files_to_check.items():
        if Path(local_path).exists():
            print(f"✅ File already exists locally: {local_path}")
        else:
            print(f"⬇️ File not found. Downloading '{minio_filename}' from MinIO...")
            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                s3_client.download_file(
                    "flood-models", minio_filename, local_path
                )
                print(f"✅ Successfully downloaded {local_path}")
            except Exception as e:
                print(f"❌ ERROR: Failed to download {minio_filename}: {e}", file=sys.stderr)
                # This is a critical failure, so we exit.
                sys.exit(1)


def run_terratorch_inference(input_dir: str, output_dir: str, input_filename: str) -> str:
    """
    Runs terratorch inference on a TIFF file. (Copied from your main.py)
    """
    predict_script = "terratorch"
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using accelerator='{accelerator}'.")

    command = [
        predict_script, "predict",
        "-c", CONFIG_PATH,
        "--ckpt_path", CHECKPOINT_PATH,
        "--predict_output_dir", output_dir,
        "--data.init_args.predict_data_root", input_dir,
        "--data.init_args.img_grep", input_filename,
        f"--trainer.accelerator={accelerator}",
        "--trainer.devices=1",
        "--data.init_args.batch_size=1",
        "--trainer.default_root_dir=/app/data"
    ]

    print(f"\nExecuting command: {' '.join(command)}")
    try:
        # Using subprocess.run for simplicity as we wait for it to complete
        result = subprocess.run(
            command,
            cwd=PROJECT_CODE_DIR,
            capture_output=True,
            text=True,
            check=True,  # This will raise an exception on non-zero exit codes
            env=os.environ.copy()
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("\nTerratorch predict command finished successfully.")

        base_name = Path(input_filename).stem
        expected_output_filename = f"{base_name}_pred.tif"
        output_filepath = Path(output_dir) / expected_output_filename

        if not output_filepath.exists():
            raise FileNotFoundError(
                f"Inference finished, but output file '{output_filepath}' was not found.")

        return str(output_filepath)

    except subprocess.CalledProcessError as e:
        print(
            f"Terratorch predict command failed with exit code {e.returncode}.", file=sys.stderr)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise gr.Error(
            f"Model inference failed. Check logs for details. STDERR: {e.stderr}")
    except Exception as e:
        print(f"An error occurred during inference: {e}", file=sys.stderr)
        raise gr.Error(f"An unexpected error occurred: {e}")


def detect_flood(image_url: str) -> str:
    """
    Performs flood detection on a GeoTIFF image provided via a URL.

    Args:
        image_url (str): A public URL pointing to a GeoTIFF image.

    Returns:
        str: The file path to the resulting prediction image, which shows flood areas.
    """
    if not image_url:
        raise gr.Error("Input is empty. Please provide a URL to a TIFF image.")

    # Create a temporary directory for processing this request
    temp_dir = tempfile.mkdtemp(prefix="flood_detect_")

    try:
        temp_dir_path = Path(temp_dir)
        input_dir = temp_dir_path / "input"
        output_dir = temp_dir_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Download the image from the URL
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        url_path = image_url.split('?')[0]
        original_filename = url_path.split('/')[-1]
        input_filename = original_filename if original_filename else "input.tif"
        input_filepath = input_dir / input_filename

        with open(input_filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Input file saved to: {input_filepath}")

        # Run the inference (this part remains the same)
        output_filepath = run_terratorch_inference(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            input_filename=input_filename
        )

        if not output_filepath:
            raise gr.Error("Inference failed to produce an output file.")

        minio_url = upload_to_minio(
            output_filepath, Path(output_filepath).name)
        return minio_url

    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from URL: {e}")
        raise gr.Error(f"Could not fetch image from URL: {image_url}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise gr.Error(str(e))


# Create the Gradio interface
demo = gr.Interface(
    fn=detect_flood,
    inputs=gr.Textbox(
        lines=1,
        placeholder="https://path/to/your/satellite_image.tif",
        label="Image URL"
    ),
    outputs=gr.Textbox(label="MinIO Result URL"),
    title="💧 Flood Detection Model 🌊",
    description="Provide a public URL to a GeoTIFF image to run flood detection."
)

# Launch the interface
if __name__ == "__main__":
    ensure_files_exist()
    demo.launch(server_name="0.0.0.0", server_port=8080, mcp_server=True)
