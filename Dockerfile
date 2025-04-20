FROM python:3.11

ARG TARGET_ENV=cpu  # Can be "cpu" or "gpu"
ENV TARGET_ENV=${TARGET_ENV}

# Copy over app dir
COPY app /app/

# Set app as working dir
WORKDIR /app


    # Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
git gcc g++ build-essential wget cmake \
libsqlite3-dev libtiff-dev libcurl4-openssl-dev sqlite3 \
&& rm -rf /var/lib/apt/lists/*

# Build and install PROJ from source (to get the right version)
RUN wget https://download.osgeo.org/proj/proj-9.2.1.tar.gz \
&& tar -xzf proj-9.2.1.tar.gz \
&& cd proj-9.2.1 \
&& mkdir build && cd build \
&& cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON \
&& make -j$(nproc) \
&& make install \
&& ldconfig \
&& cd /app \
&& rm -rf proj-9.2.1*

# Now install GDAL with the right PROJ version
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/local/share/proj
ENV LD_LIBRARY_PATH=/usr/local/lib

# Set Python path properly - define it before using it
ENV PYTHONPATH="/app"

# Ensure pip, wheel, setuptools, and numpy are installed for subsequent steps
RUN pip install --upgrade pip setuptools wheel

# Install pyproj first to ensure proper PROJ setup
RUN pip install pyproj==3.7.1

# Then install rasterio and other geospatial packages
RUN pip install --no-binary rasterio rasterio torchgeo

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
    "numpy<2" \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    rioxarray \
    mcp \
    requests

# Copy application code (ensure mcp_server.py is included)
COPY ./app /app/app
COPY ./configs /app/configs
COPY ./models /app/models
COPY mcp_server.py /app/mcp_server.py

# Create directories expected by the target structure that might not be copied
# (fine_tuning, regions, plots based on the ls output)
RUN mkdir -p \
    /app/data/input \
    /app/data/output

# Change the CMD to run the Uvicorn server
CMD ["python", "/app/mcp_server.py"]