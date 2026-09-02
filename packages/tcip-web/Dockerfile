# Run with `docker run --network host`: the backend binds the host's loopback and no port is
# published. The GUI's in-app agent terminal needs a Claude Code install this image does not carry.

FROM node:20 AS frontend-builder
WORKDIR /app
COPY packages/tcip-web/frontend/ packages/tcip-web/frontend/
WORKDIR /app/packages/tcip-web/frontend
RUN npm ci --no-fund --no-audit
RUN npm run build

FROM python:3.12-slim
WORKDIR /app

# rasterio's bundled GDAL needs libexpat; torch's OpenMP runtime needs libgomp. Neither ships
# in python:3.12-slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, the way .github/workflows/ci.yml installs it.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY packages/tcip-store/ packages/tcip-store/
COPY packages/tcip-annotation/ packages/tcip-annotation/
COPY packages/tcip-mcp/ packages/tcip-mcp/
COPY packages/tcip-web/ packages/tcip-web/

# STATIC_DIR (tcip_web/app.py) prefers a copy packaged inside the installed package; landing the
# built bundle here before the pip install below carries it into that wheel (packages/tcip-web/README.md).
COPY --from=frontend-builder /app/packages/tcip-web/static/ packages/tcip-web/src/tcip_web/static/

RUN pip install --no-cache-dir \
    packages/tcip-store \
    packages/tcip-annotation \
    packages/tcip-mcp \
    packages/tcip-web

EXPOSE 8765

CMD ["python", "-m", "tcip_web"]
