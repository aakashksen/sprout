FROM python:3.12-slim

ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SPROUT_ENV=production SPROUT_DATA_DIR=/data \
    SPROUT_MANIFEST=/app/knowledge/manifest.csv SPROUT_INDEX_DIR=/models/index \
    FASTEMBED_CACHE_PATH=/models/fastembed

WORKDIR /app
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt

COPY pyproject.toml ./
COPY src ./src
COPY deploy ./deploy
RUN sed -i 's/\r$//' deploy/entrypoint.sh deploy/wsgi.py && chmod +x deploy/entrypoint.sh
COPY knowledge/manifest.csv knowledge/manifest.csv
RUN pip install --no-deps . && rm -rf build

# /data holds the database and uploads; /models is mounted read-only and holds the ONNX model and index.
RUN useradd --system --uid 10001 --no-create-home sprout \
    && mkdir -p /data /models && chown sprout /data
USER 10001
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
