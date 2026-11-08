# Acknowledgements

GPU Shepherd is open source (Apache-2.0, see [LICENSE](LICENSE)), and it only works because of the
open-source projects below. Thank you to everyone who builds and maintains them.

The same list is in the dashboard: click the version number at the top, then **Credits**.

## How GPU Shepherd uses them

- **Built on**: Python packages GPU Shepherd installs and runs on. Only one file is bundled in this
  repository: htmx, with its license in `gpushepherd/static/htmx.LICENSE.txt`.
- **Works with** and **App Center apps**: GPU Shepherd starts these as containers, using the
  images their own publishers release (Docker Hub or GitHub's registry). It doesn't change,
  rebuild or redistribute them, and each keeps its own license. Some are not plain open-source
  licenses (Open WebUI's branding terms, MongoDB's SSPL, NVIDIA's driver): read them if you
  offer these apps to others.
- **Models**: downloaded from Hugging Face when you pick them, under the license on each model card.
  The GGUF builds are by [Unsloth](https://unsloth.ai) and Qwen; the models themselves are by their
  original makers (Qwen at Alibaba, Gemma at Google DeepMind).

## Built on

| Project | License | Used for |
| :- | :- | :- |
| [FastAPI](https://fastapi.tiangolo.com) | MIT | The web server for the dashboard and the API |
| [Uvicorn](https://www.uvicorn.org) | BSD-3-Clause | Runs the web server |
| [Jinja](https://jinja.palletsprojects.com) | BSD-3-Clause | The dashboard's pages |
| [htmx](https://htmx.org) | 0BSD | Live updates in the dashboard; bundled, with its license, in gpushepherd/static |
| [Pydantic](https://docs.pydantic.dev) | MIT | Checks and stores settings and instances |
| [PyYAML](https://pyyaml.org) | MIT | Reads and writes the YAML settings files |
| [Docker SDK for Python](https://github.com/docker/docker-py) | Apache-2.0 | Starts, stops and watches containers |
| [HTTPX](https://www.python-httpx.org) | BSD-3-Clause | Talks to model servers and checks for updates |
| [itsdangerous](https://itsdangerous.palletsprojects.com) | BSD-3-Clause | Signs login sessions |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache-2.0 | Reads form posts |
| [pypdf](https://github.com/py-pdf/pypdf) | BSD-3-Clause | Reads the text of PDF documents |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) | Apache-2.0 or BSD-3-Clause (PDFium: BSD-3-Clause) | Reads PDF text pypdf misses and draws pages for OCR |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | Reads scanned PDFs (OCR), in the container image |
| [pytest](https://pytest.org) | MIT | Tests (development only) |

## Works with

| Project | License | Used for |
| :- | :- | :- |
| [Docker Engine (Moby)](https://mobyproject.org) | Apache-2.0 | Runs every instance |
| [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) | Apache-2.0 | Gives containers their GPUs; installed by install.sh |
| [NVIDIA driver and nvidia-smi](https://www.nvidia.com/drivers) | NVIDIA's own license (not open source) | Reads the GPUs; installed by you or your distribution |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Runs GGUF models inside the Llama Swap |
| [Python](https://www.python.org) | PSF License | GPU Shepherd's own image is built on python:3.13-slim (Debian), whose packages keep their own licenses |
| [curl](https://curl.se) | curl license (MIT-style) | Downloads models, in the curlimages/curl container |
| [Tailscale](https://tailscale.com) | BSD-3-Clause (client) | Optional remote access, only if you choose it in install.sh |
| [gpu-unlock](https://github.com/Zagitta/gpu-unlock) | GPL-2.0-only (NVIDIA firmware under NVIDIA's terms) | Experimental full-speed tensor math on GeForce cards, built on your server by gpu-unlock.sh only if you install it in Settings |
| [Distil-Whisper large-v3 (faster-whisper build)](https://huggingface.co/Systran/faster-distil-whisper-large-v3) | MIT | Speaches' default speech-to-text model |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 | Speaches' default text-to-speech model (ONNX build by speaches-ai) |

## App Center apps

| App | License | Image |
| :- | :- | :- |
| [Llama Swap](https://github.com/mostlygeek/llama-swap) | MIT (llama-swap and llama.cpp) | `ghcr.io/mostlygeek/llama-swap:cuda` |
| [Ollama](https://docs.ollama.com) | MIT | `ollama/ollama:latest` |
| [vLLM](https://docs.vllm.ai) | Apache-2.0 | `vllm/vllm-openai:latest` |
| [Open WebUI](https://docs.openwebui.com) | Open WebUI License (BSD-3-Clause plus branding terms) | `ghcr.io/open-webui/open-webui:main` |
| [Hermes Agent](https://hermes-agent.nousresearch.com/docs/user-guide/docker) | MIT | `nousresearch/hermes-agent:latest` |
| [OpenCode](https://opencode.ai/docs) | MIT | `ghcr.io/anomalyco/opencode:latest` |
| [LibreChat](https://www.librechat.ai/docs) | MIT | `librechat/librechat:latest` |
| [AnythingLLM](https://docs.anythingllm.com/installation-docker/overview) | MIT | `mintplexlabs/anythingllm:latest` |
| [Langflow](https://docs.langflow.org) | MIT | `langflowai/langflow:latest` |
| [SillyTavern](https://docs.sillytavern.app) | AGPL-3.0 | `ghcr.io/sillytavern/sillytavern:latest` |
| [Open Notebook](https://github.com/lfnovo/open-notebook) | MIT | `lfnovo/open_notebook:v1-latest` |
| [InvokeAI](https://invoke.ai) | Apache-2.0 | `ghcr.io/invoke-ai/invokeai:latest` |
| [ComfyUI](https://github.com/YanWenKun/ComfyUI-Docker) | GPL-3.0 (ComfyUI), Mulan PSL v2 (the image) | `yanwk/comfyui-boot:cu130-slim-v2` |
| [Speaches](https://speaches.ai) | MIT | `ghcr.io/speaches-ai/speaches:latest-cuda` |
| [LibreChat database](https://www.librechat.ai/docs/local/docker) | SSPL v1 (MongoDB Community Server) | `mongo:8.0` |
| [Open Notebook database](https://github.com/lfnovo/open-notebook) | BSL-1.1 (SurrealDB, Apache-2.0 after four years) | `surrealdb/surrealdb:v2` |

## Suggested models

| Model | License | Download page |
| :- | :- | :- |
| Qwen3.5 9B | Apache-2.0 | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Gemma 4 E4B | Apache-2.0 | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| Gemma 4 12B | Apache-2.0 | [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) |
| Qwen3 Coder 30B-A3B | Apache-2.0 | [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) |
| DeepSeek R1 0528 Qwen3 8B | MIT | [unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF) |
| Devstral Small 2 24B | Apache-2.0 | [unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) |
| Qwen3.8 27B | Apache-2.0 | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| Muse Glimmer 30B | Apache-2.0 | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) |
| Qwen3.6 35B-A3B | Apache-2.0 | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3 Coder Next 80B-A3B | Apache-2.0 | [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) |
| Qwen3.5 122B-A10B | Apache-2.0 | [unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) |
| Ornith 1.5 122B | MIT | [unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) |
| Devstral 2 123B | Modified MIT | [unsloth/Devstral-2-123B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Devstral-2-123B-Instruct-2512-GGUF) |
| gpt-oss 120B | Apache-2.0 | [unsloth/gpt-oss-120b-GGUF](https://huggingface.co/unsloth/gpt-oss-120b-GGUF) |
| MiniMax M2.7 229B-A10B | Modified MIT | [unsloth/MiniMax-M2.7-GGUF](https://huggingface.co/unsloth/MiniMax-M2.7-GGUF) |
| Nemotron 3.5 Super VL 120B-A12B | OpenMDW-1.1 | [unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) |
| Qwen3 Embedding 0.6B | Apache-2.0 | [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) |

Spotted a missing credit or a wrong license? Please open an issue.
