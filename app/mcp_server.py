from pathlib import Path
import boto3
import os
import base64
import asyncio
from mcp.server.fastmcp import FastMCP
from uvicorn import run as uvicorn_run

import inference  # assuming this is in the same /app folder

# Initialize the MCP server
mcp = FastMCP("flood-detection")


@mcp.tool()
async def generate_flood_map(input_file_url: str, filename: str = "input.tif") -> str:
    """
    Generates a flood map from an input TIFF image located at a URL.
    The server will download the file from the provided URL.
    GPU usage is detected automatically within the container.

    Args:
        input_file_url: The publicly accessible URL of the input TIFF file.
        filename: The original filename of the input TIFF (optional, used for context/logging).

    Returns:
        A base64 encoded string of the resulting flood map TIFF image,
        or an error message string starting with 'Error:'.
    """
    print(
        f"Received request to generate flood map for file at URL: {input_file_url} (Original name: {filename})")
    try:
        # Run the inference using the URL
        # Using asyncio.to_thread for potentially blocking download and subprocess call
        output_bytes = await asyncio.to_thread(
            inference.run_terratorch_inference,  # Pass the function itself
            input_file_url,                     # Pass arguments separately
            filename
        )
        # If your inference function is already async (e.g., uses aiohttp), you can await it:
        # output_bytes = await inference.run_terratorch_inference(input_file_url, filename)

        if output_bytes:
            print(
                f"Inference successful, received {len(output_bytes)} output bytes.")
            # Encode the output bytes to base64
            output_base64 = base64.b64encode(output_bytes).decode('utf-8')
            print("Encoded output to base64.")
            return output_base64
        else:
            print("Inference failed or produced no output.")
            return "Error: Inference failed to produce an output file."

    except Exception as e:
        print(f"An unexpected error occurred in the tool: {e}")
        return f"Error: Internal server error - {e}"


def ensure_models_exist():
    """Download models from MinIO if they don't exist locally"""
    model_path = Path(
        "/app/models/granite_geospatial_uki_flood_detection_v1.ckpt")
    # config_path = Path(
    #     "/app/configs/config_granite_geospatial_uki_flood_detection_v1.yaml")

    # If models already exist, skip download
    if model_path.exists():  # and config_path.exists():
        print("Models and configs already exist locally")
        return

    print("Downloading models and configs from MinIO...")

    # MinIO configuration (from your document)
    s3_client = boto3.client(
        's3',
        endpoint_url='https://minio-s3-ppe-multi-modal.apps.cluster-r8fxn.r8fxn.sandbox753.opentlc.com',
        aws_access_key_id=os.environ.get('MINIO_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('MINIO_SECRET_KEY'),
        region_name='us-east-1'
    )

    # Create directories if they don't exist
    os.makedirs("/app/models", exist_ok=True)
    # os.makedirs("/app/configs", exist_ok=True)

    # Download model and config from MinIO
    try:
        bucket_name = 'flood-models'  # Change to your actual bucket name
        s3_client.download_file(
            bucket_name, 'granite_geospatial_uki_flood_detection_v1.ckpt', str(model_path))
        # s3_client.download_file(
        #     bucket_name, 'config_granite_geospatial_uki_flood_detection_v1.yaml', str(config_path))
        print("Successfully downloaded models and configs")
    except Exception as e:
        print(f"Error downloading models: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Starting Flood Detection MCP Server...")
    ensure_models_exist()
    # Create an ASGI app with explicitly defined paths
    # Make the root path work for SSE connection
    app = mcp.sse_app()

    # Run it with uvicorn
    uvicorn_run(app, host="0.0.0.0", port=8080)
