# The base image is pinned by digest, so a rebuild gets the same one; Dependabot moves the pin.
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS buildinfo
# Records which commit this image is built from (read from .git without needing git), so the
# dashboard can show it and tell when an update is waiting. Only the result reaches the image.
COPY . /ctx
RUN python3 /ctx/gpushepherd/buildinfo.py /ctx /build.json

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

LABEL org.opencontainers.image.title="GPU Shepherd" \
      org.opencontainers.image.description="Runs self-hosted AI containers and shares the GPUs between them automatically" \
      org.opencontainers.image.licenses="Apache-2.0"

# Tesseract reads scanned PDFs (Knowledge page, Read with OCR); add tesseract-ocr-<lang> packages
# and set SHEPHERD_OCR_LANG (e.g. eng+deu) for other languages
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# The exact versions (and file hashes) of every Python dependency, so each build installs the same
# code. Made from pyproject.toml; after changing its dependencies run:
#   pip-compile --allow-unsafe --generate-hashes --strip-extras -o requirements.txt pyproject.toml
COPY requirements.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY gpushepherd ./gpushepherd
RUN pip install --no-cache-dir --no-deps . && rm -rf /src
COPY --from=buildinfo /build.json /usr/share/gpu-shepherd/build.json
COPY CHANGELOG.md /usr/share/gpu-shepherd/CHANGELOG.md

# "image" for the image the release workflow publishes (compose.standalone.yaml): it is updated
# with docker compose pull, so the dashboard points there instead of at update.sh
ARG SHEPHERD_INSTALL=clone
ENV SHEPHERD_DATA_DIR=/opt/gpu-shepherd \
    SHEPHERD_INSTALL=${SHEPHERD_INSTALL} \
    PYTHONUNBUFFERED=1
WORKDIR /
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=3)" || exit 1
CMD ["gpu-shepherd"]
