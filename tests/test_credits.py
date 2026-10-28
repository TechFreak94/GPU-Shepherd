"""Open-source credits: every app and model names its license, and ACKNOWLEDGEMENTS.md lists them all."""
from pathlib import Path

from gpushepherd import credits, instance_templates as tpl, model_library

ACK = (Path(__file__).parent.parent / "ACKNOWLEDGEMENTS.md").read_text()


def test_every_app_and_model_has_a_license():
    assert all(t.license for t in tpl.TEMPLATES), [t.key for t in tpl.TEMPLATES if not t.license]
    assert all(m.license for m in model_library.SUGGESTIONS)
    assert all(c.license and c.url.startswith("https://") for c in credits.BUILT_ON + credits.WORKS_WITH)


def test_acknowledgements_list_everything():
    names = [c.name for c in credits.BUILT_ON + credits.WORKS_WITH] + [t.title for t in tpl.TEMPLATES] \
        + [m.name for m in model_library.SUGGESTIONS] + [t.image for t in tpl.TEMPLATES]
    missing = [n for n in names if n not in ACK]
    assert not missing, f"add these to ACKNOWLEDGEMENTS.md: {missing}"
