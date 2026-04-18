FROM python:3.11-slim

WORKDIR /app

# Install the packages
COPY packages/tcip-mcp/ packages/tcip-mcp/
COPY packages/tcip-annotation/ packages/tcip-annotation/
COPY packages/tcip-web/ packages/tcip-web/

RUN pip install --no-cache-dir \
    packages/tcip-annotation \
    packages/tcip-mcp \
    packages/tcip-web

EXPOSE 8000

CMD ["python", "-m", "tcip_web"]
