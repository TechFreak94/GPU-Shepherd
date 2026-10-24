FROM python:3.13-slim AS buildinfo
# Records which commit this image is built from (read from .git without needing git), so the
# dashboard can show it and tell when an update is waiting. Only the result reaches the image.
COPY . /ctx
RUN python3 /ctx/gpushepherd/buildinfo.py /ctx /build.json

FROM python:3.13-slim

LABEL org.opencontainers.image.title="GPU Shepherd" \
      org.opencontainers.image.description="Runs self-hosted AI containers and shares the GPUs between them automatically" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY gpushepherd ./gpushepherd
RUN pip install --no-cache-dir . && rm -rf /src
COPY --from=buildinfo /build.json /usr/share/gpu-shepherd/build.json
COPY CHANGELOG.md /usr/share/gpu-shepherd/CHANGELOG.md

ENV SHEPHERD_DATA_DIR=/opt/gpu-shepherd \
    PYTHONUNBUFFERED=1
WORKDIR /
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=3)" || exit 1
CMD ["gpu-shepherd"]
