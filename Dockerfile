FROM python:3.11

ARG TARGET_ENV=cpu  # Can be "cpu" or "gpu"
ENV TARGET_ENV=${TARGET_ENV}

# Copy over app dir
COPY app /app/

# Set app as working dir
WORKDIR /app

# Install GDAL dependencies with version specification
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc build-essential \
    gdal-bin libgdal-dev python3-gdal \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# Set Python path properly - define it before using it
ENV PYTHONPATH="/app"

# Ensure pip, wheel, setuptools, and numpy are installed for subsequent steps
RUN pip install --upgrade pip setuptools wheel

# Install dependencies directly instead of trying to install the local package
RUN pip install torch==2.1.0 torchvision==0.16.0 \
    "terratorch @ git+https://github.com/IBM/terratorch.git@6d97521" \
    lightning-utilities==0.11.3.post \
    albumentations==1.4.3 \
    huggingface_hub \
    jupyter \
    matplotlib \
    imagecodecs \
    global_land_mask \
    "numpy<2"



# Create directories expected by the target structure that might not be copied
# (fine_tuning, regions, plots based on the ls output)
RUN mkdir -p \
    /app/data/input \
    /app/data/output


ENTRYPOINT ["python", "/app/run_inference.py"]
