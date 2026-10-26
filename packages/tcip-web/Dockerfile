FROM node:22.23 AS frontend-builder
WORKDIR /app
COPY packages/tcip-web/frontend/ packages/tcip-web/frontend/
WORKDIR /app/packages/tcip-web/frontend
RUN npm ci --no-fund --no-audit && npm run build

FROM python:3.13-slim
WORKDIR /app

# rasterio's bundled GDAL needs libexpat; torch's OpenMP runtime needs libgomp; the opencv-python
# wheel the packages declare needs libGL and, for its bundled Qt, GLib. None ships in the slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY packages/tcip-store/ packages/tcip-store/
COPY packages/tcip-annotation/ packages/tcip-annotation/
COPY packages/tcip-mcp/ packages/tcip-mcp/
COPY packages/tcip-web/ packages/tcip-web/

# The builder's package source holds the built bundle alone; the install below carries it.
COPY --from=frontend-builder /app/packages/tcip-web/src/ packages/tcip-web/src/

RUN pip install --no-cache-dir \
    packages/tcip-store \
    packages/tcip-annotation \
    packages/tcip-mcp \
    packages/tcip-web

CMD ["python", "-m", "tcip_web"]
