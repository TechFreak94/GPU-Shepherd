"""Logins that end when they should, the limit on wrong logins, and files only their owner reads."""
import stat

from gpushepherd import auth
from gpushepherd.config import Store

from test_app import HX, login, stack  # noqa: F401 - the fixture


def test_a_password_change_ends_other_logins_but_not_this_one(stack):
    app, client, backend, data = stack
    login(client)
    other = dict(client.cookies)                       # a second browser, or a copied cookie
    r = client.post("/settings/password", headers=HX,
                    data={"current": "correct-horse", "new": "another-password", "again": "another-password"})
    assert r.status_code == 204
    assert client.get("/settings", follow_redirects=False).status_code == 200
    client.cookies.clear()
    client.cookies.update(other)
    assert client.get("/settings", follow_redirects=False).status_code == 303


def test_a_password_reset_ends_every_login(stack):
    app, client, backend, data = stack
    login(client)
    auth.reset_admin(Store(data))
    assert client.get("/settings", follow_redirects=False).status_code == 303


def test_a_cookie_copied_before_logout_stops_working(stack):
    app, client, backend, data = stack
    login(client)
    copied = dict(client.cookies)
    assert client.post("/logout", headers=HX).status_code == 204
    client.cookies.clear()
    client.cookies.update(copied)
    assert client.get("/settings", follow_redirects=False).status_code == 303
    assert client.get("/api/v1/settings").status_code == 401
    login(client)                                      # a new login works as before
    assert client.get("/settings", follow_redirects=False).status_code == 200


def test_wrong_logins_are_limited(stack):
    app, client, backend, data = stack
    for _ in range(auth.LoginLimiter.MAX):
        assert client.post("/login", data={"user": "admin", "password": "nope"}).status_code == 401
    r = client.post("/login", data={"user": "admin", "password": "correct-horse"}, follow_redirects=False)
    assert r.status_code == 429 and "Too many wrong logins" in r.text


def test_login_limiter_forgets_after_the_window_and_after_a_good_login():
    now = [0.0]
    lim = auth.LoginLimiter(clock=lambda: now[0])
    for _ in range(lim.MAX):
        assert lim.wait("a") == 0
        lim.failed("a")
    assert lim.wait("a") > 0 and lim.wait("b") == 0   # another address is not held up
    now[0] = lim.WINDOW + 1
    assert lim.wait("a") == 0
    lim.failed("a")
    lim.passed("a")
    assert lim._failed == {}
    for i in range(lim.MAX_ALL):                       # many addresses, a few tries each
        lim.failed(f"c{i}")
    assert lim.wait("new") > 0


def test_a_user_name_with_other_letters_is_just_wrong(stack):
    app, client, backend, data = stack
    assert client.post("/login", data={"user": "ädmin", "password": "x"}).status_code == 401


def test_back_only_takes_this_sites_paths(stack):
    app, client, backend, data = stack
    login(client)
    r = client.post("/ui/instances/webui/start", headers=HX, data={"back": "//example.com"})
    assert "example.com" not in r.headers.get("HX-Redirect", "")


def test_instance_files_are_owner_only(stack):
    app, client, backend, data = stack
    store = Store(data)
    path = store.instance_path("webui")
    path.chmod(0o644)                                  # as older versions wrote it
    assert store.tighten() >= 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    store.save_instance(store.instance("webui"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_suffix(".yaml.prev").stat().st_mode) == 0o600
