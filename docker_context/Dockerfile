# Use Python 3.10 as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install git (needed for potential editable installs) and gcc (for compiling C extensions)
RUN apt-get update && apt-get install -y git gcc build-essential gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*

# Copy the specific project code (uki-flooddetection) into the container
# Adjust the source path if your cloned repo isn't named 'geospatial'
COPY ../geospatial/uki-flooddetection /app/uki-flooddetection

# Copy the inference script into the container
COPY docker_context/run_inference.py /app/run_inference.py

# Install the project dependencies using its setup.py
# This installs terratorch and other required libraries
WORKDIR /app/uki-flooddetection
RUN pip install --no-cache-dir .

# Set the main workdir back to /app
WORKDIR /app

# Define the entrypoint for the container
# We run the python script which then calls terratorch
ENTRYPOINT ["python", "/app/run_inference.py"]