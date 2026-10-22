# Example: the "aistack" setup

ComfyUI pinned to GPU 0, an elastic llama-swap server with three models (and the auto names `auto` and `coder`), Hermes Agent and Open WebUI.
`gpu-shepherd --demo` starts with these files. To use them for real, copy them into your data folder
(`/opt/gpu-shepherd/instances` and `/opt/gpu-shepherd/catalogs`), then fix the `/home/you/...` paths,
the Open WebUI API key and the images in the UI.
