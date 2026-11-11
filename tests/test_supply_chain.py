"""What gets installed and pulled is pinned: the image's Python packages, the helper image, the Actions."""
import re
import tomllib
from pathlib import Path

from gpushepherd.backend import split_image
from gpushepherd.downloads import IMAGE

ROOT = Path(__file__).parent.parent


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def test_requirements_pin_every_dependency_with_hashes():
    """requirements.txt (what the Dockerfile installs) is made from pyproject.toml: see the Dockerfile."""
    wanted = {_name(d) for d in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]}
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    pinned = {_name(x) for x in lines if re.match(r"^[A-Za-z0-9]", x)}
    assert wanted <= pinned, f"not in requirements.txt: {sorted(wanted - pinned)}"
    assert all("==" in x for x in lines if re.match(r"^[A-Za-z0-9]", x))
    assert sum("--hash=sha256:" in x for x in lines) >= len(pinned)


def test_the_helper_image_and_the_base_image_are_pinned_by_digest():
    assert re.search(r"@sha256:[0-9a-f]{64}$", IMAGE)
    froms = re.findall(r"^FROM (\S+)", (ROOT / "Dockerfile").read_text(), re.M)
    assert froms and all(re.search(r"@sha256:[0-9a-f]{64}$", f) for f in froms)


def test_actions_are_pinned_by_commit():
    for f in (ROOT / ".github" / "workflows").glob("*.yml"):
        for ref in re.findall(r"uses: (\S+)", f.read_text()):
            assert re.search(r"@[0-9a-f]{40}$", ref), f"{f.name}: {ref}"


def test_split_image():
    assert split_image("mongo:8.0") == ("mongo", "8.0")
    assert split_image("ghcr.io/open-webui/open-webui") == ("ghcr.io/open-webui/open-webui", "latest")
    assert split_image("registry.local:5000/app") == ("registry.local:5000/app", "latest")
    assert split_image("registry.local:5000/app:1.2") == ("registry.local:5000/app", "1.2")
    d = "sha256:" + "a" * 64
    assert split_image(f"curlimages/curl@{d}") == ("curlimages/curl", d)
    assert split_image(f"curlimages/curl:8.11.1@{d}") == ("curlimages/curl", d)
    assert split_image(f"registry.local:5000/app@{d}") == ("registry.local:5000/app", d)
