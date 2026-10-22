#!/usr/bin/env python3
"""Turn a folder of generated PNG/JPG/WEBP coloring pages into a print-ready PDF.

Each page is: converted to grayscale -> scaled (Lanczos) to fit inside the safe
area of a 300-DPI page -> tone-curved so the background is pure white and lines
are solid black -> centered on a white page. The PDF is built losslessly (img2pdf).

Default page = 8.5 x 11 in, 0.5 in margins, no bleed.
Needs Pillow and img2pdf. With GPU Shepherd, run it through kdp-book.sh next to it, which
runs it in a throwaway container so nothing has to be installed:
    /opt/gpu-shepherd/appdata/comfyui/tools/kdp-book.sh \
        /opt/gpu-shepherd/appdata/comfyui/output/my-book ~/my_book.pdf --blank-backs
"""
import argparse
import sys
from pathlib import Path

import img2pdf
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="folder of page images (sorted by file name)")
    ap.add_argument("-o", "--output", type=Path, default=Path("book.pdf"))
    ap.add_argument("--width", type=float, default=8.5, help="trim width, inches")
    ap.add_argument("--height", type=float, default=11.0, help="trim height, inches")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.5, help="safe margin, inches")
    ap.add_argument("--low", type=int, default=100, help="gray level mapped to pure black")
    ap.add_argument("--high", type=int, default=200, help="gray level mapped to pure white")
    ap.add_argument("--bw", action="store_true", help="hard 1-bit black/white pages (smallest file)")
    ap.add_argument("--blank-backs", action="store_true", help="insert a blank page after every page")
    args = ap.parse_args()

    files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        return 1

    page_w, page_h = round(args.width * args.dpi), round(args.height * args.dpi)
    box_w, box_h = page_w - 2 * round(args.margin * args.dpi), page_h - 2 * round(args.margin * args.dpi)

    span = max(1, args.high - args.low)
    lut = [0 if v <= args.low else 255 if v >= args.high else round((v - args.low) * 255 / span) for v in range(256)]
    cut = (args.low + args.high) // 2

    pages_dir = args.output.with_suffix("")
    pages_dir = pages_dir.parent / (pages_dir.name + "_pages")
    pages_dir.mkdir(parents=True, exist_ok=True)

    def save_page(img: Image.Image, idx: int) -> str:
        if args.bw:
            img = img.point(lambda v: 255 if v > cut else 0).convert("1", dither=Image.Dither.NONE)
        path = pages_dir / f"page_{idx:03d}.png"
        img.save(path, dpi=(args.dpi, args.dpi))
        return str(path)

    out_pages, n = [], 0
    blank = Image.new("L", (page_w, page_h), 255)
    for f in files:
        with Image.open(f) as im:
            g = im.convert("L")
        scale = min(box_w / g.width, box_h / g.height)
        g = g.resize((max(1, round(g.width * scale)), max(1, round(g.height * scale))), Image.Resampling.LANCZOS)
        g = g.point(lut)
        canvas = Image.new("L", (page_w, page_h), 255)
        canvas.paste(g, ((page_w - g.width) // 2, (page_h - g.height) // 2))
        n += 1
        out_pages.append(save_page(canvas, n))
        if args.blank_backs:
            n += 1
            out_pages.append(save_page(blank, n))

    args.output.write_bytes(img2pdf.convert(out_pages))
    print(f"Wrote {args.output}  ({len(out_pages)} pages, {args.width}x{args.height} in @ {args.dpi} DPI)")
    print(f"Page PNGs for review: {pages_dir}")
    if len(out_pages) < 24:
        print("Note: KDP paperbacks need at least 24 pages.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
