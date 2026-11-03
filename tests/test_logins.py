"""App logins: which apps GPU Shepherd knows, the scripts it runs in them, and reading their output."""
import pytest

from gpushepherd import logins
from gpushepherd.config import ConfigError, Instance


def inst(app, **env):
    return Instance(id="x", name="X", image="x", app=app, env=env)


def test_methods():
    assert logins.method(inst("comfyui")) is None and logins.method(inst("")) is None
    assert not logins.method(inst("hermes")).in_container
    old = Instance(id="x", name="X", image="ghcr.io/open-webui/open-webui:main")       # from before the App Center
    assert logins.method(old).where == "its database"
    for app in ("open-webui", "librechat", "langflow", "anythingllm"):
        m = logins.method(inst(app))
        assert m.in_container and logins.script(m).cmd
    with pytest.raises(ConfigError):
        logins.script(logins.method(inst("hermes")))


def test_scripts_take_the_password_from_their_environment():
    for app, workdir in (("librechat", "/app/config"), ("anythingllm", "/app/server"), ("langflow", "")):
        ex = logins.script(logins.method(inst(app)), "me@x.y", "s3cret-pass")
        assert ex.env == {"GS_USER": "me@x.y", "GS_PASSWORD": "s3cret-pass"} and ex.workdir == workdir
        assert "s3cret-pass" not in " ".join(ex.cmd)
    assert logins.script(logins.method(inst("open-webui"))).env == {}


def test_readable():
    h = inst("hermes", HERMES_DASHBOARD_BASIC_AUTH_USERNAME="admin", HERMES_DASHBOARD_BASIC_AUTH_PASSWORD="pw")
    assert logins.readable(logins.method(h), h, None) == [logins.Account("admin", "dashboard", "pw")]
    a = inst("anythingllm")
    assert logins.readable(logins.method(a), a, "VECTOR_DB='lancedb'\n") == []
    assert logins.readable(logins.method(a), a, "AUTH_TOKEN=\"tok\"\n")[0].password == "tok"


def test_env_file():
    text = "A='1'\nAUTH_TOKEN='old'\n"
    assert logins.env_value(text, "AUTH_TOKEN") == "old" and logins.env_value(text, "B") == ""
    new = logins.set_env_value(text, "AUTH_TOKEN", "new")
    assert new == "A='1'\nAUTH_TOKEN='new'\n"
    assert logins.set_env_value(new, "JWT_SECRET", "s").endswith("JWT_SECRET='s'\n")


def test_parse():
    accounts, multi = logins.parse_accounts('warming up\n[{"user": "a@b.c", "role": "admin"}, {"user": ""}]')
    assert accounts == [logins.Account("a@b.c", "admin")] and multi
    accounts, multi = logins.parse_accounts('{"multi_user": false, "users": []}')
    assert accounts == [] and not multi
    with pytest.raises(ConfigError, match="boom"):
        logins.parse_accounts("Error: boom")
    assert logins.parse_updated('{"updated": 1}') == 1
    with pytest.raises(ConfigError):
        logins.parse_updated("")


def test_check_password():
    logins.check_password("long-enough")
    for bad in ("short", "has'quote", "two\nlines"):
        with pytest.raises(ConfigError):
            logins.check_password(bad)
