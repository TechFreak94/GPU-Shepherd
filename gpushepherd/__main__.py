"""`gpu-shepherd` command: runs the web UI, or with a command (status, start, apps, models...)
manages the one that is running (cli.py)."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn

from . import __version__, cli

PRESETS = Path(__file__).resolve().parent / "presets"


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="gpu-shepherd", description="GPU Shepherd: without a command, runs the web UI; with one, manages the "
        "GPU Shepherd that is running (the same things the dashboard does).",
        epilog="Options go before the command. See docs/cli.md, or `gpu-shepherd COMMAND -h`.")
    ap.add_argument("--host", default=os.environ.get("SHEPHERD_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHEPHERD_PORT", "8090")))
    ap.add_argument("--data-dir", default=os.environ.get("SHEPHERD_DATA_DIR", "/opt/gpu-shepherd"))
    ap.add_argument("--demo", action="store_true",
                    help="no Docker needed: fake GPUs and containers, seeded with a preset")
    ap.add_argument("--demo-preset", default="starter", metavar="NAME",
                    help="preset the demo starts with (a folder in gpushepherd/presets; default: starter)")
    ap.add_argument("--reset-password", action="store_true",
                    help="give the admin a new password, print it and exit (the running web UI uses it at once)")
    ap.add_argument("--version", action="version", version=f"gpu-shepherd {__version__}")
    ap.add_argument("--url", default="", metavar="URL",
                    help="commands: the GPU Shepherd to manage (default: http://127.0.0.1:PORT, or $SHEPHERD_URL)")
    ap.add_argument("--token", default="", metavar="TOKEN",
                    help="commands: its API token (default: $SHEPHERD_TOKEN, or the api-token file in the data folder)")
    ap.add_argument("--json", action="store_true", help="commands: print the API's JSON answer")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    cli.add_commands(sub)
    args = ap.parse_args()
    if args.command:
        raise SystemExit(cli.run(args))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = args.data_dir
    if args.reset_password:
        from . import auth
        from .config import Store
        store = Store(data_dir)
        user, password = auth.reset_admin(store)
        print(f"New login for GPU Shepherd\n  user: {user}\n  password: {password}\n"
              f"Also saved in {auth.initial_password_file(store)}. Change it under Settings.")
        return
    if args.demo:
        os.environ["SHEPHERD_DEMO"] = "1"
        os.environ.setdefault("SHEPHERD_FAKE_GPUS", "2x8192:NVIDIA GeForce RTX 3070")
        os.environ.setdefault("SHEPHERD_ADMIN_PASSWORD", "demo-password")
        data_dir = tempfile.mkdtemp(prefix="gpushepherd-demo-")
        preset = PRESETS / args.demo_preset
        if not (preset / "instances").is_dir():
            ap.error(f"no preset named '{args.demo_preset}' in {PRESETS}")
        for part in ("instances", "catalogs", "appdata"):
            if (preset / part).is_dir():
                shutil.copytree(preset / part, Path(data_dir) / part, dirs_exist_ok=True)
        logging.getLogger("gpushepherd").warning("Demo mode: log in as admin / demo-password (data in %s)", data_dir)

    from .app import create_app
    uvicorn.run(create_app(data_dir=data_dir), host=args.host, port=args.port, log_level="info",
                proxy_headers=False)


if __name__ == "__main__":
    main()
