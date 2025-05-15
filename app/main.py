from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
import subprocess
import boto3
from pathlib import Path
from api import endpoints
import sys
import torch

app = FastAPI(title="Flood Detection API")

app.include_router(endpoints.router, prefix="/api/v1")

# MinIO configuration
MINIO_ENDPOINT = 'https://minio-s3-ppe-multi-modal.apps.cluster-r8fxn.r8fxn.sandbox753.opentlc.com'
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
MINIO_BUCKET = 'flood-predictions'

# Define paths
CONFIG_PATH = '/app/configs/config_granite_geospatial_uki_flood_detection_v1.yaml'
CHECKPOINT_PATH = '/app/models/granite_geospatial_uki_flood_detection_v1.ckpt'
PROJECT_CODE_DIR = "/app"  # Directory where terratorch command should run


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Flood Detection API"}


async def run_terratorch_inference(
    input_dir: str,
    output_dir: str,
    input_filename: str
) -> str:
    """
    Runs terratorch inference on a TIFF file and returns the path to the output file.
    
    Args:
        input_dir: Directory containing the input file
        output_dir: Directory where output will be saved
        input_filename: Name of the input file
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    predict_script = "terratorch"
    
    # Detect GPU availability
    accelerator = 'cpu'
    devices = 1
    try:
        if torch.cuda.is_available():
            accelerator = 'gpu'
            print("✅ GPU detected. Using accelerator='gpu'.")
        else:
            print("ℹ️ No GPU detected or PyTorch CUDA not available. Using accelerator='cpu'.")
    except Exception as e:
        print(f"⚠️ Error during GPU detection: {e}. Defaulting to CPU.", file=sys.stderr)
    
    # Construct the command
    command = [
        predict_script,
        "predict",
        "-c", CONFIG_PATH,
        "--ckpt_path", CHECKPOINT_PATH,
        "--predict_output_dir", output_dir,
        "--data.init_args.predict_data_root", input_dir,
        # Use the specific input filename
        "--data.init_args.img_grep", input_filename,
        f"--trainer.accelerator={accelerator}",
        f"--trainer.devices={devices}",
        "--data.init_args.batch_size=1",
        "--trainer.default_root_dir=/app/data"
    ]
    
    print(f"\nExecuting command: {' '.join(command)}\nWorking directory: {PROJECT_CODE_DIR}\n")
    
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_CODE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy()
        )
        
        # Stream output for logging/debugging
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)
        
        rc = process.poll()
        
        if rc == 0:
            print("\nTerratorch predict command finished successfully.")
            
            # Determine the output filename
            base_name = Path(input_filename).stem
            expected_output_filename = f"{base_name}_pred.tif"
            output_filepath = Path(output_dir) / expected_output_filename
            
            if not output_filepath.exists():
                output_files = list(Path(output_dir).glob('*.tif*'))
                if output_files:
                    print(
                        f"Warning: Expected output '{expected_output_filename}' not found. "
                        f"Using first found TIF: {output_files[0].name}")
                    output_filepath = output_files[0]
                else:
                    print(f"Error: No output TIF file found in {output_dir}", file=sys.stderr)
                    return None
            
            return str(output_filepath)
        else:
            print(f"\nTerratorch predict command failed with exit code {rc}.", file=sys.stderr)
            return None
    
    except FileNotFoundError:
        print(
            f"Error: Command '{predict_script}' not found. Is terratorch installed and in PATH?",
            file=sys.stderr
        )
        return None
    except Exception as e:
        print(f"An error occurred during inference execution: {e}", file=sys.stderr)
        return None


def upload_to_minio(file_path: str, object_name: str) -> str:
    """
    Upload a file to MinIO storage.
    
    Args:
        file_path: Path to the file to upload
        object_name: Name to give the file in MinIO
        
    Returns:
        URL to the uploaded file in MinIO
    """
    try:
        # Initialize MinIO client
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name='us-east-1'
        )
        
        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=MINIO_BUCKET)
        except Exception:
            s3_client.create_bucket(Bucket=MINIO_BUCKET)
            print(f"Created bucket: {MINIO_BUCKET}")
        
        # Upload the file
        print(f"Uploading {file_path} to MinIO bucket {MINIO_BUCKET} as {object_name}")
        s3_client.upload_file(file_path, MINIO_BUCKET, object_name)
        
        # Generate URL for accessing the file
        url = f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_name}"
        print(f"File uploaded. URL: {url}")
        
        return url
    
    except Exception as e:
        print(f"Error uploading to MinIO: {e}", file=sys.stderr)
        raise e


@app.post("/predict-stream/")
async def predict_flood_streaming(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input TIFF file for flood detection.")
):
    """
    Endpoint that streams a TIFF file for model prediction and stores the output in MinIO.
    
    This endpoint:
    1. Takes an uploaded TIFF file
    2. Streams it for processing by the terratorch model
    3. Stores the output in MinIO storage
    4. Returns the prediction result URL
    
    Returns:
        A JSON object with the MinIO URL of the prediction result
    """
    if not file.filename.lower().endswith(".tif") and not file.filename.lower().endswith(".tiff"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only .tif or .tiff files are accepted.")
    
    # Create a temporary directory for this request
    temp_dir = tempfile.mkdtemp(prefix="flood_detect_stream_")
    temp_dir_path = Path(temp_dir)
    input_dir = temp_dir_path / "input"
    output_dir = temp_dir_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    input_filepath = input_dir / file.filename
    print(f"Input file path is: {input_filepath}")
    
    # Clean up temp directory function
    def cleanup_temp_dir(temp_dir_path: Path):
        """Removes the temporary directory."""
        try:
            shutil.rmtree(temp_dir_path)
            print(f"Cleaned up temporary directory: {temp_dir_path}")
        except Exception as e:
            print(f"Error cleaning up temp directory {temp_dir_path}: {e}")
    
    try:
        # Stream file to temporary storage
        with open(input_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Input file saved to: {input_filepath}")
        
        # Run terratorch inference
        output_filepath = await run_terratorch_inference(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            input_filename=file.filename
        )
        
        if not output_filepath or not Path(output_filepath).exists():
            raise HTTPException(status_code=500, detail="Inference failed to produce output file")
        
        # Upload result to MinIO
        minio_url = upload_to_minio(output_filepath, Path(output_filepath).name)
        
        # Add cleanup task
        # background_tasks.add_task(cleanup_temp_dir, temp_dir_path)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Flood detection completed successfully",
            "result_url": minio_url,
            "filename": Path(output_filepath).name
        })
    
    except HTTPException as http_exc:
        # Ensure cleanup happens even if there's an HTTP exception before response
        # cleanup_temp_dir(temp_dir_path)
        raise http_exc
    except Exception as e:
        # Ensure cleanup happens on any unexpected error
        # cleanup_temp_dir(temp_dir_path)
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # Close the uploaded file explicitly
        await file.close()


@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    # Ensure MinIO credentials are available
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        print("WARNING: MinIO credentials not found in environment. /predict-stream/ endpoint may not work.")


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
