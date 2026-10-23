#!/usr/bin/env bash
# Z-Image Turbo on GPU Shepherd: one ComfyUI per GPU, set up for coloring-book pages.
#
# The GPU Shepherd version of setup_comfyui_zimage.sh. Instead of installing ComfyUI on the host
# with systemd services, it runs install.sh with the zimage-2x8gb preset (a ComfyUI container per
# GPU, 8188 and 8189, plus the LLM server, Hermes and Open WebUI) and then downloads:
#   - Z-Image Turbo (6B, Apache-2.0): Kijai's fp8-scaled build sized for 8 GB cards (or the
#     official bf16 build), the Qwen3-4B text encoder and the VAE
#   - the Z-Image coloring-book LoRA (artificialguybr, Apache-2.0)
#   - the RealESRGAN anime-6B upscaler (good for line art)
#
# Usage (as a normal user with sudo; any install.sh option is passed on, e.g. --yes):
#   ./install-zimage.sh
#   ZIMAGE_VARIANT=both ./install-zimage.sh     # fp8 + official bf16 (+ both text encoders)
#   INSTALL_LLMS=1 ./install-zimage.sh          # also the preset's chat and coding models (~72 GB)
#   HF_TOKEN=hf_xxx ./install-zimage.sh         # only if a download says 401/403
#
# Safe to re-run: install.sh only does what is missing and downloads skip files you have. The
# ComfyUI containers need NVIDIA driver 580 or newer (CUDA 13).
set -euo pipefail

ZIMAGE_VARIANT="${ZIMAGE_VARIANT:-fp8}"    # fp8 (6 GB, best for 8 GB cards) | bf16 (12 GB, official) | both
INSTALL_LORAS="${INSTALL_LORAS:-1}"        # Z-Image coloring-book LoRA
INSTALL_UPSCALER="${INSTALL_UPSCALER:-1}"  # RealESRGAN anime-6B
INSTALL_LLMS="${INSTALL_LLMS:-0}"          # the preset's LLMs: Qwen3.6 35B-A3B and Qwen3 Coder Next
HF_TOKEN="${HF_TOKEN:-}"
PRESET=zimage-2x8gb

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
case "$ZIMAGE_VARIANT" in fp8|bf16|both) ;; *) die "ZIMAGE_VARIANT must be fp8, bf16 or both" ;; esac

if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null || die "run this as root"
    exec sudo -- env ZIMAGE_VARIANT="$ZIMAGE_VARIANT" INSTALL_LORAS="$INSTALL_LORAS" \
        INSTALL_UPSCALER="$INSTALL_UPSCALER" INSTALL_LLMS="$INSTALL_LLMS" HF_TOKEN="$HF_TOKEN" \
        "$HERE/$(basename "$0")" "$@"
fi

# GPU Shepherd itself, with the preset. Its own download offer is turned off: the Z-Image files
# are fetched below, by variant.
status=0
"$HERE/install.sh" --preset "$PRESET" --no-download "$@" || status=$?
case "$status" in
    0) ;;
    2) exit 2 ;;                                     # the plan was declined; nothing changed
    3) say "After the reboot, run this again:  ./install-zimage.sh"; exit 3 ;;
    *) exit "$status" ;;
esac

# Where install.sh put things (compose.yaml keeps the data and models folders).
compose="$HERE/compose.yaml"
[ -f "$compose" ] || die "compose.yaml not found; did install.sh finish?"
data_dir=$(sed -n 's/^ *SHEPHERD_DATA_DIR: *//p' "$compose" | head -n 1)
models_dir=$(sed -n 's#^ *- \(/[^:]*\):\1:ro$#\1#p' "$compose" | head -n 1)
[ -n "$models_dir" ] || die "compose.yaml has no models folder; add one (see compose.example.yaml) and run this again"
if [ ! -f "$data_dir/instances/comfyui.yaml" ]; then
    warn "$data_dir already had instances, so the preset was not added. The models are downloaded"
    warn "anyway; add the ComfyUI instances from gpushepherd/presets/$PRESET/instances by hand."
fi

groups=()
case "$ZIMAGE_VARIANT" in fp8) groups+=(zimage) ;; bf16) groups+=(zimage-bf16) ;; both) groups+=(zimage zimage-bf16) ;; esac
[ "$INSTALL_LORAS" != 1 ] || groups+=(lora)
[ "$INSTALL_UPSCALER" != 1 ] || groups+=(upscaler)
[ "$INSTALL_LLMS" != 1 ] || groups+=(chat coder)

echo
say "Downloading into $models_dir: ${groups[*]}"
as=()
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then as=(sudo -u "$SUDO_USER"); fi
failed=0
"${as[@]}" env HF_TOKEN="$HF_TOKEN" bash "$HERE/gpushepherd/presets/download-models.sh" \
    "$PRESET" "$models_dir" "${groups[@]}" || failed=1

host_ip=$(hostname -I 2>/dev/null | awk '{print $1}'); host_ip=${host_ip:-<server-ip>}
fp8_note=""
[ "$ZIMAGE_VARIANT" = bf16 ] || fp8_note="
    With the fp8 build, pick z-image-turbo_fp8_scaled_e4m3fn_KJ (Load Diffusion Model)
    and qwen_3_4b_fp8_mixed (Load CLIP) in its loaders."
cat <<EOF

  ComfyUI, GPU 0:     http://$host_ip:8188   (images in $data_dir/appdata/comfyui/output)
  ComfyUI, GPU 1:     http://$host_ip:8189   (images in $data_dir/appdata/comfyui-2/output;
                      starting it stops the LLM server, GPU Shepherd asks first)
  Start them on the GPU Shepherd dashboard. The first start downloads the ComfyUI image.

  Start here: Workflow > Browse Templates > "Z-Image-Turbo: Text to Image".$fp8_note
    1024x1328, 8 steps, CFG 1.
  Coloring-book LoRA: add LoraLoaderModelOnly after Load Diffusion Model with
    zimage/ColoringBook_Redmond_ZImage.safetensors; prompts start "Coloring Book, ColoringBookAF".

  SECURITY: ComfyUI has no login, and custom nodes can run code. Keep it on your LAN.
EOF
if [ "$failed" -eq 1 ]; then
    warn "some downloads failed. If the error was 401/403, make a read token at"
    warn "https://huggingface.co/settings/tokens, then run:  HF_TOKEN=hf_xxx ./install-zimage.sh"
    exit 1
fi
