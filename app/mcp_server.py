import gradio as gr
import os
import base64
import tempfile
# import shutil
import subprocess
from pathlib import Path
import torch

# Define fixed paths from your container setup
CONFIG_PATH = '/app/configs/config_granite_geospatial_uki_flood_detection_v1.yaml'
CHECKPOINT_PATH = '/app/models/granite_geospatial_uki_flood_detection_v1.ckpt'
PROJECT_CODE_DIR = "/app"

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
            raise FileNotFoundError(f"Inference finished, but output file '{output_filepath}' was not found.")
            
        return str(output_filepath)

    except subprocess.CalledProcessError as e:
        print(f"Terratorch predict command failed with exit code {e.returncode}.", file=sys.stderr)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise gr.Error(f"Model inference failed. Check logs for details. STDERR: {e.stderr}")
    except Exception as e:
        print(f"An error occurred during inference: {e}", file=sys.stderr)
        raise gr.Error(f"An unexpected error occurred: {e}")


def detect_flood(base64_tiff_string: str) -> str:
    """
    Performs flood detection on a base64 encoded GeoTIFF image.

    Args:
        base64_tiff_string (str): A base64 encoded string of a GeoTIFF image. It can be prefixed with a data URI like 'data:image/tiff;base64,'.

    Returns:
        str: The file path to the resulting prediction image, which shows flood areas.
    """
    if not base64_tiff_string:
        raise gr.Error("Input is empty. Please provide a base64 encoded TIFF string.")

    # Create a temporary directory for processing this request
    temp_dir = tempfile.mkdtemp(prefix="flood_detect_")
    
    try:
        temp_dir_path = Path(temp_dir)
        input_dir = temp_dir_path / "input"
        output_dir = temp_dir_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Decode the base64 string and save as a .tif file
        # This handles strings with or without the 'data:image/tiff;base64,' prefix
        if "," in base64_tiff_string:
            _, encoded = base64_tiff_string.split(",", 1)
        else:
            encoded = base64_tiff_string

        image_bytes = base64.b64decode(encoded)
        input_filename = "input.tif"
        input_filepath = input_dir / input_filename

        with open(input_filepath, "wb") as f:
            f.write(image_bytes)
        
        print(f"Input file saved to: {input_filepath}")

        # Run the inference
        output_filepath = run_terratorch_inference(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            input_filename=input_filename
        )

        if not output_filepath:
            raise gr.Error("Inference failed to produce an output file.")

        # Gradio's gr.Image will handle this filepath and display the image.
        # The temporary file will be automatically served by Gradio.
        return output_filepath

    except Exception as e:
        # Ensure cleanup happens on any error
        print(f"An error occurred: {e}")
        raise gr.Error(str(e))
    # Note: Gradio manages the cleanup of temporary files returned by functions.
    # If you didn't return the path, you would use a finally block:
    # finally:
    #     shutil.rmtree(temp_dir)


# Create the Gradio interface
demo = gr.Interface(
    fn=detect_flood,
    inputs=gr.Textbox(
        lines=5, 
        placeholder="Paste your base64 encoded TIFF image string here...",
        label="Base64 Input TIFF"
    ),
    outputs=gr.Image(
        type="filepath", 
        label="Flood Detection Result"
    ),
    title="💧 Flood Detection Model 🌊",
    description="Provide a base64 encoded GeoTIFF image to run flood detection. The model will output an image mask showing predicted flood areas."
)

# Launch the interface
if __name__ == "__main__":
    # The `launch()` function starts the web server.
    # Set server_name to "0.0.0.0" to make it accessible outside the container.
    demo.launch(server_name="0.0.0.0", server_port=8080, mcp_server=True)