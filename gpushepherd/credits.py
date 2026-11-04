"""Open-source credits: the projects GPU Shepherd is built on and works with, and their licenses.

Shown in the dashboard's Credits dialog and written out in ACKNOWLEDGEMENTS.md (a test keeps the
two in step). App Center apps and suggested models carry their own license in instance_templates.py
and model_library.py; this module lists the rest.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Credit:
    name: str
    url: str
    license: str
    use: str               # what GPU Shepherd uses it for


# Python packages GPU Shepherd itself runs on (pyproject.toml), and the one file it bundles
BUILT_ON: tuple[Credit, ...] = (
    Credit("FastAPI", "https://fastapi.tiangolo.com", "MIT", "The web server for the dashboard and the API"),
    Credit("Uvicorn", "https://www.uvicorn.org", "BSD-3-Clause", "Runs the web server"),
    Credit("Jinja", "https://jinja.palletsprojects.com", "BSD-3-Clause", "The dashboard's pages"),
    Credit("htmx", "https://htmx.org", "0BSD", "Live updates in the dashboard; bundled, with its license, "
           "in gpushepherd/static"),
    Credit("Pydantic", "https://docs.pydantic.dev", "MIT", "Checks and stores settings and instances"),
    Credit("PyYAML", "https://pyyaml.org", "MIT", "Reads and writes the YAML settings files"),
    Credit("Docker SDK for Python", "https://github.com/docker/docker-py", "Apache-2.0",
           "Starts, stops and watches containers"),
    Credit("HTTPX", "https://www.python-httpx.org", "BSD-3-Clause", "Talks to model servers and checks for updates"),
    Credit("itsdangerous", "https://itsdangerous.palletsprojects.com", "BSD-3-Clause", "Signs login sessions"),
    Credit("python-multipart", "https://github.com/Kludex/python-multipart", "Apache-2.0", "Reads form posts"),
    Credit("pypdf", "https://github.com/py-pdf/pypdf", "BSD-3-Clause", "Reads the text of PDF books"),
    Credit("pytest", "https://pytest.org", "MIT", "Tests (development only)"),
)

# what runs beside it: the platform it drives and the images and tools it pulls or installs
WORKS_WITH: tuple[Credit, ...] = (
    Credit("Docker Engine (Moby)", "https://mobyproject.org", "Apache-2.0", "Runs every instance"),
    Credit("NVIDIA Container Toolkit", "https://github.com/NVIDIA/nvidia-container-toolkit", "Apache-2.0",
           "Gives containers their GPUs; installed by install.sh"),
    Credit("NVIDIA driver and nvidia-smi", "https://www.nvidia.com/drivers", "NVIDIA's own license (not open source)",
           "Reads the GPUs; installed by you or your distribution"),
    Credit("llama.cpp", "https://github.com/ggml-org/llama.cpp", "MIT",
           "Runs GGUF models inside the LLM server"),
    Credit("Python", "https://www.python.org", "PSF License", "GPU Shepherd's own image is built on python:3.13-slim "
           "(Debian), whose packages keep their own licenses"),
    Credit("curl", "https://curl.se", "curl license (MIT-style)",
           "Downloads models, in the curlimages/curl container"),
    Credit("Tailscale", "https://tailscale.com", "BSD-3-Clause (client)",
           "Optional remote access, only if you choose it in install.sh"),
    Credit("gpu-unlock", "https://github.com/Zagitta/gpu-unlock", "GPL-2.0-only (NVIDIA firmware under NVIDIA's terms)",
           "Experimental full-speed tensor math on GeForce cards, built on your server by gpu-unlock.sh only if you "
           "install it in Settings"),
    Credit("Distil-Whisper large-v3 (faster-whisper build)", "https://huggingface.co/Systran/faster-distil-whisper-large-v3",
           "MIT", "Speaches' default speech-to-text model"),
    Credit("Kokoro-82M", "https://huggingface.co/hexgrad/Kokoro-82M", "Apache-2.0",
           "Speaches' default text-to-speech model (ONNX build by speaches-ai)"),
)

NOTE = ("App Center apps and models are downloaded from their own publishers when you install them; GPU "
        "Shepherd doesn't change or redistribute them, and each keeps its own license. Thank you to everyone "
        "who builds and maintains these projects.")
