"""`gpu-shepherd` command: runs the web UI."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn

from . import __version__

EXAMPLES = Path(__file__).resolve().parent / "examples"


def main() -> None:
    ap = argparse.ArgumentParser(prog="gpu-shepherd", description="GPU Shepherd web UI")
    ap.add_argument("--host", default=os.environ.get("SHEPHERD_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHEPHERD_PORT", "8090")))
    ap.add_argument("--data-dir", default=os.environ.get("SHEPHERD_DATA_DIR", "/opt/gpu-shepherd"))
    ap.add_argument("--demo", action="store_true",
                    help="no Docker needed: fake GPUs and containers, seeded with the example stack")
    ap.add_argument("--version", action="version", version=f"gpu-shepherd {__version__}")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = args.data_dir
    if args.demo:
        os.environ["SHEPHERD_DEMO"] = "1"
        os.environ.setdefault("SHEPHERD_FAKE_GPUS", "2x8192:NVIDIA GeForce RTX 3070")
        os.environ.setdefault("SHEPHERD_ADMIN_PASSWORD", "demo-password")
        data_dir = tempfile.mkdtemp(prefix="gpushepherd-demo-")
        example = EXAMPLES / "aistack"
        if example.is_dir():
            shutil.copytree(example, data_dir, dirs_exist_ok=True)
        logging.getLogger("gpushepherd").warning("Demo mode: log in as admin / demo-password (data in %s)", data_dir)

    from .app import create_app
    uvicorn.run(create_app(data_dir=data_dir), host=args.host, port=args.port, log_level="info",
                proxy_headers=False)


if __name__ == "__main__":
    main()
