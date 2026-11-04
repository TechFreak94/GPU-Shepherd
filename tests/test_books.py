"""Books: finding and reading ebooks, sending them to Open WebUI, and skills made from them."""
import shutil
import time
import zipfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, book_skills, books
from gpushepherd.app import create_app
from gpushepherd.backend import FakeBackend
from gpushepherd.config import Store
from gpushepherd.connectors import openwebui

EXAMPLE = Path(__file__).parent / "fixtures" / "stack"
WORDS = " ".join(f"word{i}" for i in range(300))


def epub(path: Path, title="A Book", author="Ann Author", chapters=None, drm=False) -> Path:
    chapters = chapters or [("One", f"First chapter. {WORDS}"), ("Two", f"Second chapter. {WORDS} end")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        items = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(len(chapters)))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        z.writestr("OEBPS/content.opf", f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title>'
                   f'<dc:creator>{author}</dc:creator></metadata><manifest>{items}</manifest><spine>{spine}</spine></package>')
        for i, (head, body) in enumerate(chapters):
            z.writestr(f"OEBPS/c{i}.xhtml", f"<html><head><title>x</title><style>p{{}}</style></head><body>"
                       f"<h1>{head}</h1><p>{body}</p><ul><li>step a</li><li>step b</li></ul></body></html>")
        if drm:
            z.writestr("META-INF/encryption.xml", '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                       'xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData>'
                       '<enc:EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#aes128-cbc"/>'
                       '<enc:CipherData><enc:CipherReference URI="OEBPS/c0.xhtml"/></enc:CipherData>'
                       '</enc:EncryptedData></encryption>')
    return path


def pdf(path: Path, text: str) -> Path:
    """A one-page PDF with text in it."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    w = PdfWriter()
    page = w.add_blank_page(612, 792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(
        {NameObject("/F1"): w._add_object(font)})})
    lines = [text[i:i + 80] for i in range(0, len(text), 80)]
    ops = "BT /F1 8 Tf 20 770 Td 10 TL " + " ".join(f"({x}) Tj T*" for x in lines) + " ET"
    stream = DecodedStreamObject()
    stream.set_data(ops.encode())
    page[NameObject("/Contents")] = w._add_object(stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        w.write(f)
    return path


# --------------------------------------------------------------------------- reading

def test_epub_text_title_and_drm(tmp_path):
    title, author, text = books.read(epub(tmp_path / "a.epub"))
    assert (title, author) == ("A Book", "Ann Author")
    assert text.index("# One") < text.index("# Two") and "- step a" in text and "p{}" not in text
    with pytest.raises(books.BookError, match="DRM"):
        books.read(epub(tmp_path / "b.epub", drm=True))
    with pytest.raises(books.BookError, match="Calibre"):
        books.read((tmp_path / "c.mobi"))


def test_pdf_text(tmp_path):
    title, _, text = books.read(pdf(tmp_path / "p.pdf", WORDS))
    assert "word10" in text and "word299" in text
    blank = tmp_path / "blank.pdf"
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(3):
        w.add_blank_page(612, 792)
    w.write(str(blank))
    with pytest.raises(books.BookError, match="OCR"):
        books.read(blank)


def test_find_groups_copies_of_one_book(tmp_path):
    (tmp_path / "bundle" / "deep").mkdir(parents=True)
    for f in ("deep-learning.epub", "deep-learning.pdf", "deep-learning.mobi"):
        (tmp_path / "bundle" / "deep" / f).write_text("x")
    (tmp_path / "calibre").mkdir()
    for f in ("Title - Author.azw3", "Title - Author retail.pdf"):
        (tmp_path / "calibre" / f).write_text("x")
    for f in ("one.pdf", "two.pdf", "notes.txt"):                 # different books side by side
        (tmp_path / f).write_text("x")
    found = dict(books.find(tmp_path))
    assert found["bundle/deep/deep-learning.epub"] == ["bundle/deep/deep-learning.mobi", "bundle/deep/deep-learning.pdf"]
    assert found["calibre/Title - Author retail.pdf"] == ["calibre/Title - Author.azw3"]   # read from a copy it can convert
    assert {"one.pdf", "two.pdf", "notes.txt"} <= set(found)


def test_scan_converts_skips_and_finds_duplicates(tmp_path):
    lib = books.Library(tmp_path)
    folder = tmp_path / "books"
    epub(folder / "Bundle A" / "book.epub")
    epub(folder / "Bundle B" / "same book renamed.epub", title="Another Title")    # the same text
    epub(folder / "locked.epub", drm=True)
    (folder / "old.mobi").write_text("x")
    (folder / "notes.md").write_text("# Notes\n\n" + WORDS)
    counts = lib.scan()
    assert counts == {"ready": 2, "duplicate": 1, "failed": 1, "skipped": 1}
    rows = {b.path: b for b in lib.books()}
    assert rows["Bundle B/same book renamed.epub"].status == "duplicate"
    assert rows["Bundle A/book.epub"].group == "Bundle A" and rows["Bundle A/book.epub"].name == "A Book - Ann Author"
    assert "DRM" in rows["locked.epub"].reason
    text = lib.text(rows["Bundle A/book.epub"].key)
    assert text.startswith("# A Book\n\nBy Ann Author") and "word5" in text
    # unchanged books aren't read again, and keep where they were sent
    lib.update(lambda d: d["books"][rows["notes.md"].key].sent.update({"webui/Books": "f1"}))
    assert lib.scan() == counts
    assert {b.path: b for b in lib.books()}["notes.md"].sent == {"webui/Books": "f1"}
    # a removed book is forgotten, with its text
    (folder / "notes.md").unlink()
    lib.scan()
    assert "notes.md" not in {b.path for b in lib.books()} and not lib.text_path(rows["notes.md"].key).exists()


# --------------------------------------------------------------------------- skills

def test_skill_parts_and_chapters():
    text = "# Title\n\n" + "\n\n".join(f"## Chapter {i}\n\n" + WORDS for i in range(5))
    ps = book_skills.parts(text, 700)
    assert all(len(p.split()) <= 700 for p in ps) and sum(len(p.split()) for p in ps) == len(text.split())
    assert [t for t, _ in book_skills.chapters(text)] == ["Front matter"] + [f"Chapter {i}" for i in range(5)]
    assert book_skills.chapters("no headings") == [("Book", "no headings")]
    assert book_skills.part_words(8192) == int((8192 - 3000) / 1.4) and book_skills.part_words(0) == 12000


def test_make_skill_with_a_model():
    asked = []

    def chat(prompt):
        asked.append(prompt)
        if "SKILL.md" in prompt:
            return "<think>hmm</think>```markdown\n---\nname: x\ndescription: Use when \"baking\" bread.\n---\n\n# Bread\n\n## How to do it\n1. Knead\n```"
        return "NOTHING" if "PART 2 " in prompt else "- knead for 10 minutes"
    text = "\n\n".join(WORDS for _ in range(10))
    md = book_skills.make("Bread Book - Ann", text, chat, ctx=4500)
    assert md.startswith("---\nname: bread-book-ann\ndescription: \"Use when 'baking' bread.\"\n---\n\n# Bread")
    assert sum("--- PART" in p for p in asked) == len(asked) - 1
    with pytest.raises(ValueError):
        book_skills.make("x", text, lambda p: "NOTHING", ctx=0)


# --------------------------------------------------------------------------- the service, API and pages

class FakeOpenWebUI:
    made: dict = {}
    uploads: list = []
    embedding_now = {"engine": "", "model": "sentence-transformers/all-MiniLM-L6-v2", "url": ""}
    switched: list = []

    def __init__(self, base, token):
        assert token == "tok" and base.endswith(":8080") or base.endswith(":3000")

    def embedding(self):
        return self.embedding_now

    def use_embedding(self, url, key, model, query_prefix="", batch=16):
        self.switched.append((url, model))

    def knowledge_id(self, name, description=""):
        return self.made.setdefault(name, f"kb{len(self.made)}")

    def upload(self, name, text):
        if "Broken" in name:
            raise openwebui.OpenWebUIError("Open WebUI answered 500: nope")
        self.uploads.append(name)
        return f"f{len(self.uploads)}"

    def add(self, kid, fid):
        pass


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_FAKE_GPUS", "2x8192:Test GPU")
    monkeypatch.setenv("SHEPHERD_ADMIN_PASSWORD", "correct-horse")
    monkeypatch.setattr(gpu_mod, "_cache", (0.0, [], ""))
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    h = tmp_path / "instances" / "hermes.yaml"
    h.write_text(h.read_text().replace("/opt/gpu-shepherd", str(tmp_path)))
    for name in ("webui", "hermes"):                          # as the App Center installs them
        f = tmp_path / "instances" / f"{name}.yaml"
        data = yaml.safe_load(f.read_text())
        data["app"] = {"webui": "open-webui", "hermes": "hermes"}[name]
        f.write_text(yaml.safe_dump(data))
    backend = FakeBackend()
    backend.exec_reply = lambda name, cmd, env: (0, 'noise\n{"token": "tok"}')
    app = create_app(data_dir=str(tmp_path), backend=backend, demo=True, autostart=False)
    svc = app.state.svc
    FakeOpenWebUI.made, FakeOpenWebUI.uploads, FakeOpenWebUI.switched = {}, [], []
    svc.openwebui_client = FakeOpenWebUI
    token = auth.api_token(Store(tmp_path))
    client = TestClient(app, headers={"Authorization": f"Bearer {token}"})
    return app, client, backend, tmp_path


def browser(app):
    b = TestClient(app)
    b.post("/login", data={"user": "admin", "password": "correct-horse"})
    return b


def start(app, client, iid):
    assert client.post(f"/api/v1/instances/{iid}/start").status_code in (200, 202)
    for _ in range(500):
        if not app.state.jobs.busy:
            return
        time.sleep(0.01)
    raise AssertionError("start did not finish")


def finish(svc):
    for _ in range(500):
        if svc.book_task and not svc.book_task.running:
            return svc.book_task
        time.sleep(0.01)
    raise AssertionError("book work did not finish")


def test_send_books_to_open_webui(stack):
    app, client, backend, data = stack
    svc = app.state.svc
    epub(data / "books" / "Cooking" / "bread.epub", title="Bread")
    epub(data / "books" / "Cooking" / "broken.epub", title="Broken", chapters=[("X", "other " + WORDS)])
    epub(data / "books" / "Code" / "py.epub", title="Python", chapters=[("Y", "python " + WORDS)])
    r = client.post("/api/v1/books/send", json={"instance": "webui"})
    assert r.status_code == 409 and "Start" in r.json()["error"]
    start(app, client, "webui")
    assert "No books ready" in client.post("/api/v1/books/send", json={"instance": "webui"}).json()["error"]
    assert client.post("/api/v1/books/scan").status_code == 202
    assert finish(svc).result.startswith("Found the books: 3 ready")
    assert client.post("/api/v1/books/send", json={"instance": "webui", "name": "Library", "split": True}).status_code == 202
    t = finish(svc)
    assert not t.error, t.error
    assert sorted(FakeOpenWebUI.made) == ["Library - Code", "Library - Cooking"]
    assert sorted(FakeOpenWebUI.uploads) == ["Bread - Ann Author.md", "Python - Ann Author.md"]
    assert "1 failed" in t.result and "Broken" in t.errors[0]
    # again: only what isn't there yet
    client.post("/api/v1/books/send", json={"instance": "webui", "name": "Library", "split": True})
    assert finish(svc).result.startswith("Sent 0 books") and len(FakeOpenWebUI.uploads) == 2
    listed = client.get("/api/v1/books").json()
    assert listed["sends"][0]["name"] == "Library" and listed["webuis"][0]["running"]
    page = browser(app).get("/books").text
    assert "Knowledge for Open WebUI" in page and "In webui: Library - Cooking" in page


def test_send_asks_before_changing_open_webuis_embeddings(stack, monkeypatch):
    app, client, backend, data = stack
    svc = app.state.svc
    start(app, client, "webui")
    epub(data / "books" / "bread.epub")
    svc.scan_books()
    finish(svc)
    from gpushepherd import instance_templates as tpl
    m = svc.machine()
    m.servers["llm"]["embed"] = "embed"
    monkeypatch.setattr(svc, "machine", lambda: m)
    client.post("/api/v1/books/send", json={"instance": "webui"})
    assert "Choose which to use" in finish(svc).error and not FakeOpenWebUI.uploads
    client.post("/api/v1/books/send", json={"instance": "webui", "embeddings": "server"})
    assert not finish(svc).error
    assert FakeOpenWebUI.switched == [(tpl._server_url("llm", m), "embed")]


def test_upload_folder_and_cli(stack, capsys):
    app, client, backend, data = stack
    svc = app.state.svc
    path = epub(data / "x" / "up.epub")
    with open(path, "rb") as f:
        r = client.post("/api/v1/books/upload", files=[("file", ("up.epub", f)), ("file", ("evil.exe", b"x"))])
    assert r.status_code == 202 and "left out evil.exe" in r.json()["message"]
    finish(svc)
    assert (data / "books" / "up.epub").exists() and not (data / "books" / "evil.exe").exists()
    assert "can't see" in client.put("/api/v1/books/folder", json={"path": "/nope/books"}).json()["error"]
    assert client.put("/api/v1/books/folder", json={"path": str(data / "x")}).status_code == 200
    assert client.get("/api/v1/books").json()["folder"] == str(data / "x")


def test_bulk_upload_keeps_folders_unpacks_zips_and_skips(stack, monkeypatch):
    app, client, backend, data = stack
    svc = app.state.svc
    book = epub(data / "x" / "a.epub").read_bytes()
    r = client.put("/api/v1/books/files", params={"path": "Bundle One/Cooking/a.epub"}, content=book)
    assert r.status_code == 200 and r.json()["added"] == ["Bundle One/Cooking/a.epub"]
    zpath = data / "x" / "Bundle Two.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("Python/b.epub", epub(data / "x" / "b.epub", title="Python").read_bytes())
        z.writestr("readme.html", "<p>hi</p>")
        z.writestr("../evil.epub", "x")
        z.writestr("__MACOSX/Python/._b.epub", "x")
    r = client.put("/api/v1/books/files", params={"path": "Bundle Two.zip"}, content=zpath.read_bytes())
    assert r.json() == {"added": ["Bundle Two/Python/b.epub"], "left_out": 2}
    folder = data / "books"
    assert (folder / "Bundle Two" / "Python" / "b.epub").exists() and not (folder / "Bundle Two.zip").exists()
    assert not (data / "evil.epub").exists() and not list(folder.rglob("*.part"))
    assert client.get("/api/v1/books/files").json()["files"] == {
        "Bundle One/Cooking/a.epub": len(book), "Bundle Two/Python/b.epub": (data / "x" / "b.epub").stat().st_size}
    # refused: outside the folder, not a book, too big, a zip with no books
    for path in ("../up.epub", "/etc/x/../../up.epub", "a/.hidden/b.epub", "run.exe", ""):
        assert client.put("/api/v1/books/files", params={"path": path}, content=b"x").status_code == 400, path
    monkeypatch.setattr(books, "UPLOAD_MB", 0)
    r = client.put("/api/v1/books/files", params={"path": "big.epub"}, content=b"x" * 10)
    assert r.status_code == 413 and not (folder / "big.epub").exists() and not (folder / "big.epub.part").exists()
    monkeypatch.setattr(books, "UPLOAD_MB", 200)
    empty = data / "x" / "none.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("notes.html", "x")
    assert "no books" in client.put("/api/v1/books/files", params={"path": "none.zip"}, content=empty.read_bytes()).json()["error"]
    assert "can't open" in client.put("/api/v1/books/files", params={"path": "bad.zip"}, content=b"nope").json()["error"]
    # the dashboard's route: signed in and from the page only
    b = browser(app)
    assert b.put("/books/files", params={"path": "c.txt"}, content=WORDS.encode()).status_code == 403
    r = b.put("/books/files", params={"path": "Notes/c.txt"}, content=WORDS.encode(), headers={"HX-Request": "true"})
    assert r.json()["added"] == ["Notes/c.txt"]
    assert "Notes/c.txt" in b.get("/books/files").json()["files"]
    assert 'id="book-drop"' in b.get("/books").text
    assert client.post("/api/v1/books/scan").status_code == 202
    assert not finish(svc).error
    assert {x.path for x in svc.library.books()} >= {"Bundle One/Cooking/a.epub", "Bundle Two/Python/b.epub"}


def test_cli_upload_sends_a_folder_and_skips_what_is_there(stack, tmp_path, capsys):
    from gpushepherd import cli
    app, client, backend, data = stack

    class Via(cli.Client):           # the CLI's client, over the test app instead of the network
        def __init__(self):
            self.url = ""

        def call(self, method, path, body=None, ok=(200, 201, 202)):
            r = client.request(method, path, json=body)
            assert r.status_code in ok, r.text
            return r.json()

        def send_file(self, path, file):
            r = client.put(path, content=file.read_bytes())
            if r.status_code >= 300:
                raise cli.Failed(r.json()["error"])
            return r.json()

    lib = tmp_path / "out" / "Library"
    epub(lib / "Bread" / "bread.epub", title="Bread")
    (lib / "Bread" / "cover.jpg").write_bytes(b"x")
    (lib / ".cache").mkdir()
    (lib / ".cache" / "x.epub").write_bytes(b"x")
    cli.upload_books(Via(), [lib])
    assert (data / "books" / "Library" / "Bread" / "bread.epub").exists()
    assert not (data / "books" / "Library" / ".cache").exists()
    assert "Added 1 book," in capsys.readouterr().out
    finish(app.state.svc)
    cli.upload_books(Via(), [lib])
    assert "Added 0 books, skipped 1 already there." in capsys.readouterr().out


def test_skill_made_checked_and_installed(stack, monkeypatch):
    app, client, backend, data = stack
    svc = app.state.svc
    epub(data / "books" / "bread.epub", title="Bread")
    svc.scan_books()
    finish(svc)
    key = svc.library.books()[0].key
    assert "Start" in client.post(f"/api/v1/books/{key}/skill", json={"server": "llm"}).json()["error"]
    start(app, client, "llm")
    monkeypatch.setattr(svc, "_chat", lambda server, m: lambda p: (
        "---\nname: x\ndescription: Use when baking.\n---\n\n# Bread\n1. Knead" if "SKILL.md" in p else "- knead"))
    assert client.post(f"/api/v1/books/{key}/skill", json={"server": "llm"}).status_code == 202
    t = finish(svc)
    assert not t.error, t.error
    slug = "bread-ann-author"
    draft = data / "book-skills" / slug
    assert (draft / "SKILL.md").read_text().startswith(f"---\nname: {slug}\n")
    assert sorted(p.name for p in (draft / "references").iterdir()) == ["01-front-matter.md", "02-one.md", "03-two.md"]
    assert "1. Knead" in client.get(f"/api/v1/skills/{slug}").json()["text"]
    bad = client.put(f"/api/v1/skills/{slug}", json={"text": "# no frontmatter"})
    assert bad.status_code == 400
    assert client.put(f"/api/v1/skills/{slug}", json={"text": f"---\nname: {slug}\ndescription: Use when.\n---\n# Edited"}).status_code == 200
    assert "isn't Hermes" in client.post(f"/api/v1/skills/{slug}/install", json={"instance": "webui"}).json()["error"]
    assert client.post(f"/api/v1/skills/{slug}/install", json={"instance": "hermes"}).status_code == 200
    installed = data / "appdata" / "hermes" / "skills" / "books" / slug / "SKILL.md"
    assert installed.read_text().endswith("# Edited\n")
    assert client.get("/api/v1/books").json()["skills"][0]["installed"].keys() == {"hermes"}
    page = browser(app).get("/books").text
    assert "Install in Hermes" in page and "Make a skill" in page
    assert client.delete(f"/api/v1/skills/{slug}").status_code == 200
    assert not draft.exists() and installed.exists()


def test_open_webui_client(monkeypatch):
    calls = []

    def request(method, url, headers=None, timeout=None, **kw):
        calls.append((method, url.removeprefix("http://w:8080"), kw))
        path = url.removeprefix("http://w:8080")
        reply = {"/api/v1/knowledge/?page=1": {"items": [{"id": "k1", "name": "A"}], "total": 2},
                 "/api/v1/knowledge/?page=2": {"items": [{"id": "k2", "name": "B"}], "total": 2},
                 "/api/v1/knowledge/create": {"id": "k3"},
                 "/api/v1/files/?process=true&process_in_background=false": {"id": "f1"},
                 "/api/v1/files/f1/process/status": {"status": "completed"},
                 "/api/v1/retrieval/embedding": {"RAG_EMBEDDING_ENGINE": "", "RAG_EMBEDDING_MODEL": "mini",
                                                 "openai_config": {"url": "", "key": ""}}}
        if path == "/api/v1/knowledge/k1/file/add":
            return httpx.Response(400, json={"detail": "Duplicate content detected. Please provide unique content."})
        if path == "/api/v1/knowledge/k2/file/add":
            return httpx.Response(400, json={"detail": "Upload limit reached"})
        return httpx.Response(200, json=reply.get(path, {}))
    import httpx
    monkeypatch.setattr(httpx, "request", request)
    ow = openwebui.OpenWebUI("http://w:8080/", "tok")
    assert ow.knowledge_bases() == {"A": "k1", "B": "k2"}
    assert ow.knowledge_id("B") == "k2" and ow.knowledge_id("C") == "k3"
    assert ow.upload("x.md", "text") == "f1"
    assert calls[-2][2]["files"]["file"][0] == "x.md"
    ow.add("k1", "f1")                                     # already there: fine
    with pytest.raises(openwebui.OpenWebUIError, match="Upload limit"):
        ow.add("k2", "f1")
    assert ow.embedding() == {"engine": "", "model": "mini", "url": ""}
    ow.use_embedding("http://llm:8080/v1", "sk", "embed", "Query: ")
    sent = calls[-1][2]["json"]
    assert sent["RAG_EMBEDDING_MODEL"] == sent["embedding_model"] == "embed" and sent["RAG_EMBEDDING_QUERY_PREFIX"] == "Query: "
