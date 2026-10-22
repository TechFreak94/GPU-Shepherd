# Coloring-book page cheat sheet (Z-Image Turbo)

## First run
Open the UI -> Templates (Browse Templates) -> **Z-Image-Turbo: Text to Image**.

With the default fp8 build (what download-models.sh gets), change two dropdowns in that template:
- **Load Diffusion Model** -> `z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors`
- **Load CLIP**            -> `qwen_3_4b_fp8_mixed.safetensors`
(The VAE is `ae.safetensors`. With the `zimage-bf16` download, the template's defaults already match.)

Leave the template's step count (8) and CFG (1) alone - the model is distilled for that.
A negative prompt has no effect at CFG 1, so describe what you WANT, positively.

## Canvas
- Set width/height in the latent node to **1024 x 1328** (about 8.5:11; keep multiples of 16). Batch size 1.
- Optional: "Upscale Image (using Model)" with RealESRGAN_x4plus_anime_6B for more pixels
  before the KDP step. The KDP helper also rescales and cleans the background.

## Coloring-book LoRA
1. Add a **LoraLoaderModelOnly** node right after Load Diffusion Model; pick
   `zimage/ColoringBook_Redmond_ZImage.safetensors`.
2. Trigger words in the prompt: `Coloring Book, ColoringBookAF`.
3. Start at strength 1.0; lower it (try 0.7) if the style looks overcooked.
   Compare with and without the LoRA on a few subjects - Z-Image may not need it.

## Prompt skeleton (keep the style sentence identical across a book; change only the subject)
`Coloring Book, ColoringBookAF. Black and white coloring book page, {SUBJECT}, bold clean
uniform black outlines, no shading, no gray tones, no filled black areas, large open areas to
color, plain white background, centered composition, whole subject visible with empty margin
around it.`

Complexity dials (put in the subject text):
- Toddlers: "very simple shapes, thick lines, few details"
- Kids 4-8: "simple, medium line thickness, a few background elements"
- Adults: "intricate, detailed patterns, fine line work"

Z-Image renders English text well, so you can try titles/lettering for cover art
(check each result; this is untested for line-art pages).

## Two GPUs = two independent ComfyUI servers
- "ComfyUI" on GPU 0: port 8188 -> appdata/comfyui/output
- "ComfyUI (GPU 1)": port 8189 -> appdata/comfyui-2/output. Starting it stops the LLM server
  (GPU Shepherd asks first); start the LLM server again when you are done.
- Open both in separate tabs and queue work on each (roughly 2x throughput).
- The model fits on one 8 GB card, so splitting it across GPUs would not make anything faster.

## Build the book
    /opt/gpu-shepherd/appdata/comfyui/tools/kdp-book.sh \
        ~/my-book-pages ~/my_book.pdf --blank-backs
Copy only the pages you want into one folder first (files are ordered by name).
Defaults: 8.5x11 in, 300 DPI, 0.5 in margins, no bleed. Check the KDP print previewer before publishing.
