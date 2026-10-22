#!/usr/bin/env bash
# Downloads the model files the example setup uses into your models folder.
#
#   ./download-models.sh /srv/models                 # everything except zimage-bf16 (~83 GB)
#   ./download-models.sh /srv/models zimage lora     # only some groups
#   HF_TOKEN=hf_xxx ./download-models.sh ...         # only if a download says 401/403
#
# Groups (sizes are approximate):
#   chat         Qwen3.6 35B-A3B, the LLM server's "chat" model                  ~22 GB
#   coder        Qwen3 Coder Next 80B-A3B, "coder" when the LLM has both GPUs     ~49 GB
#   zimage       Z-Image Turbo for ComfyUI: fp8 model, fp8 text encoder, VAE      ~12 GB
#   zimage-bf16  the full-precision Z-Image Turbo model and text encoder          ~20 GB
#   lora         the coloring-book LoRA for Z-Image (loras/zimage/)               <1 GB
#   upscaler     RealESRGAN x4plus anime 6B (upscale_models/)                     <1 GB
#
# Downloads resume where they stopped and files you already have are skipped, so run it again
# after a failure. Run it as yourself, not root, so the files are yours.
set -uo pipefail

ALL_GROUPS="chat coder zimage zimage-bf16 lora upscaler"
DEFAULT_GROUPS="chat coder zimage lora upscaler"
HF_TOKEN="${HF_TOKEN:-}"
FAILED=()

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
    ''|-h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; [ -n "${1:-}" ]; exit $? ;;
esac
MODELS="${1%/}"; shift
[ -d "$MODELS" ] || die "$MODELS does not exist"
[ -w "$MODELS" ] || die "you can't write to $MODELS (run it as the folder's owner)"
groups="${*:-$DEFAULT_GROUPS}"
for g in $groups; do
    case " $ALL_GROUPS " in *" $g "*) ;; *) die "unknown group '$g' (groups: $ALL_GROUPS)" ;; esac
done
want() { case " $groups " in *" $1 "*) return 0 ;; esac; return 1; }
command -v curl >/dev/null || die "curl is not installed"

# get <dest> <url> [fallback-url ...]: resumable (.part file); returns 1 if every URL failed.
# -g because one file name has [ ] in it, which curl would otherwise read as a pattern.
get() {
    local dest="$1" url auth; shift
    mkdir -p "$(dirname "$dest")"
    if [ -s "$dest" ]; then say "Already have ${dest#"$MODELS"/}"; return 0; fi
    for url in "$@"; do
        auth=()
        if [ -n "$HF_TOKEN" ] && [[ "$url" == https://huggingface.co/* ]]; then
            auth=(-H "Authorization: Bearer $HF_TOKEN")
        fi
        say "Downloading ${dest#"$MODELS"/}"
        if curl -g -fL --retry 5 --retry-delay 5 --retry-connrefused -C - --progress-bar \
                "${auth[@]}" -o "$dest.part" "$url"; then
            mv -f "$dest.part" "$dest"
            return 0
        fi
        warn "failed: $url"
    done
    return 1
}
# like get, but notes the failure for the summary
dl() { get "$@" || FAILED+=("${1#"$MODELS"/}"); }

# first *.safetensors file in a Hugging Face repo, URL-encoded (nothing on failure)
hf_first_safetensors() {
    command -v python3 >/dev/null || return 0
    curl -fsSL "https://huggingface.co/api/models/$1" 2>/dev/null | python3 -c '
import sys, json, urllib.parse
try:
    files = [s["rfilename"] for s in json.load(sys.stdin).get("siblings", [])]
except Exception:
    files = []
print(urllib.parse.quote(next((f for f in files if f.endswith(".safetensors")), "")))
' || true
}

C="$MODELS/comfyui"
ZI="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files"

if want chat; then
    dl "$MODELS/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
        https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
fi
if want coder; then
    dl "$MODELS/Qwen3-Coder-Next-UD-Q4_K_XL.gguf" \
        https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
fi

# Z-Image Turbo. fp8 (Kijai's community build) fits an 8 GB card; if it can't be downloaded,
# fall back to the official bf16 files, which ComfyUI offloads to system RAM as needed.
bf16_model=0; bf16_te=0
if want zimage; then
    get "$C/diffusion_models/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors" \
        https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors \
        || { warn "the fp8 model failed; getting the bf16 one instead"; bf16_model=1; }
    get "$C/text_encoders/qwen_3_4b_fp8_mixed.safetensors" "$ZI/text_encoders/qwen_3_4b_fp8_mixed.safetensors" \
        || { warn "the fp8 text encoder failed; getting the bf16 one instead"; bf16_te=1; }
    dl "$C/vae/ae.safetensors" "$ZI/vae/ae.safetensors"
fi
if want zimage-bf16; then bf16_model=1; bf16_te=1; fi
[ "$bf16_model" -eq 0 ] || dl "$C/diffusion_models/z_image_turbo_bf16.safetensors" "$ZI/diffusion_models/z_image_turbo_bf16.safetensors"
[ "$bf16_te" -eq 0 ] || dl "$C/text_encoders/qwen_3_4b.safetensors" "$ZI/text_encoders/qwen_3_4b.safetensors"
if want zimage-bf16 && ! want zimage; then
    dl "$C/vae/ae.safetensors" "$ZI/vae/ae.safetensors"
fi

if want lora; then
    # The file is named "[ZImage.Turbo]ColoringBook_Redmond.safetensors"; if it is renamed upstream,
    # take the repository's first .safetensors file instead.
    repo=https://huggingface.co/artificialguybr/COLORINGBOOK-REDMOND-ZIMAGE/resolve/main
    dest="$C/loras/zimage/ColoringBook_Redmond_ZImage.safetensors"
    if ! get "$dest" "$repo/%5BZImage.Turbo%5DColoringBook_Redmond.safetensors"; then
        other=$(hf_first_safetensors artificialguybr/COLORINGBOOK-REDMOND-ZIMAGE)
        if [ -z "$other" ] || ! get "$dest" "$repo/$other"; then FAILED+=("${dest#"$MODELS"/}"); fi
    fi
fi

if want upscaler; then
    dl "$C/upscale_models/RealESRGAN_x4plus_anime_6B.pth" \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    warn "these could not be downloaded (run this again to retry; a 401/403 needs HF_TOKEN):"
    printf '    %s\n' "${FAILED[@]}" >&2
    exit 1
fi
say "All done: $groups"
if want zimage || want zimage-bf16; then
    echo "    In ComfyUI: Workflow > Browse Templates > \"Z-Image-Turbo: Text to Image\", then pick the"
    echo "    files above in its loaders. Good start: 1024x1328, 8 steps, CFG 1."
fi
if want lora; then
    echo "    Coloring-book LoRA: add a LoRA loader (loras/zimage/) and start prompts with"
    echo "    \"Coloring Book, ColoringBookAF\"."
fi
