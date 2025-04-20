# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ build-essential wget cmake \
    libsqlite3-dev libtiff-dev libcurl4-openssl-dev sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Build and install PROJ from source
WORKDIR /tmp
RUN wget https://download.osgeo.org/proj/proj-9.2.1.tar.gz \
    && tar -xzf proj-9.2.1.tar.gz \
    && cd proj-9.2.1 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp \
    && rm -rf proj-9.2.1* proj-9.2.1.tar.gz

# Final stage
FROM python:3.11-slim

# Copy PROJ libraries from builder stage
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/include/ /usr/local/include/
COPY --from=builder /usr/local/share/proj/ /usr/local/share/proj/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Install runtime dependencies AND build dependencies needed for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin libgomp1 \
    gcc g++ build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/local/share/proj
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH="/app"

# Install Python dependencies in a single layer with cache cleaning
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pyproj==3.7.1 && \
    pip install --no-cache-dir --no-binary rasterio rasterio torchgeo && \
    pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    "terratorch @ git+https://github.com/IBM/terratorch.git@6d97521" \
    lightning-utilities==0.11.3.post \
    albumentations==1.4.3 \
    huggingface_hub \
    matplotlib \
    imagecodecs \
    global_land_mask \
    "numpy<2" \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    rioxarray \
    mcp \
    requests \
    boto3

# Clean up build dependencies to reduce image size
RUN apt-get update && apt-get remove -y gcc g++ build-essential git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directory structure
WORKDIR /app
RUN mkdir -p /app/data/input /app/data/output /app/configs /app/models

# Copy only necessary files
COPY app /app/app/
COPY mcp_server.py /app/

# Set the command
CMD ["python", "/app/mcp_server.py"]