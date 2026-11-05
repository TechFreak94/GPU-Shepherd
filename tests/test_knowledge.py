"""Books: finding and reading ebooks, sending them to Open WebUI, and skills made from them."""
import shutil
import time
import zipfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import gpushepherd.gpu as gpu_mod
from gpushepherd import auth, knowledge, knowledge_skills
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
    title, author, text = knowledge.read(epub(tmp_path / "a.epub"))
    assert (title, author) == ("A Book", "Ann Author")
    assert text.index("# One") < text.index("# Two") and "- step a" in text and "p{}" not in text
    with pytest.raises(knowledge.BookError, match="DRM"):
        knowledge.read(epub(tmp_path / "b.epub", drm=True))
    with pytest.raises(knowledge.BookError, match="Calibre"):
        knowledge.read((tmp_path / "c.mobi"))


def test_pdf_text(tmp_path):
    title, _, text = knowledge.read(pdf(tmp_path / "p.pdf", WORDS))
    assert "word10" in text and "word299" in text
    blank = tmp_path / "blank.pdf"
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(3):
        w.add_blank_page(612, 792)
    w.write(str(blank))
    with pytest.raises(knowledge.BookError, match="OCR"):
        knowledge.read(blank)


def test_pdf_with_empty_password_aes_and_a_broken_page(tmp_path, monkeypatch):
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(clone_from=str(pdf(tmp_path / "p.pdf", WORDS)))
    w.add_blank_page(612, 792)
    w.encrypt(user_password="", owner_password="shop", algorithm="AES-128")   # how many shops sell PDFs
    w.write(str(tmp_path / "aes.pdf"))
    assert "word299" in knowledge.read(tmp_path / "aes.pdf")[2]
    real = PdfReader.pages.fget

    class Pages(list):
        def __getitem__(self, i):
            page = list.__getitem__(self, i)
            if i == 1:
                page.extract_text = lambda: 1 / 0               # pypdf fails on an odd page
            return page
    monkeypatch.setattr(PdfReader, "pages", property(lambda r: Pages(real(r))))
    assert "word299" in knowledge.read(tmp_path / "aes.pdf")[2]


def test_pdf_page_text_joins_hyphens_keeps_lines():
    assert knowledge.pdf_page_text("An exam-\nple  of   text  \n\n\n\ndef x():\n    return 1\x00") == \
        "An example of text\n\ndef x():\n    return 1"


def test_zip_has_books(tmp_path):
    def mk(name, files):
        with zipfile.ZipFile(tmp_path / name, "w") as z:
            for n, data in files.items():
                z.writestr(n, data)
        return tmp_path / name
    assert knowledge.zip_has_books(mk("music.zip", {"a/01.mp3": "x", "a/readme.txt": "short", "__MACOSX/a/._b.pdf": "x"})) is False
    assert knowledge.zip_has_books(mk("book.zip", {"a/b.pdf": "x", "c.jpg": "x"})) is True
    assert knowledge.zip_has_books(mk("long.zip", {"notes.md": "word " * 5000})) is True
    (tmp_path / "bad.zip").write_bytes(b"nope")
    assert knowledge.zip_has_books(tmp_path / "bad.zip") is None


def test_find_groups_copies_of_one_book(tmp_path):
    (tmp_path / "bundle" / "deep").mkdir(parents=True)
    for f in ("deep-learning.epub", "deep-learning.pdf", "deep-learning.mobi"):
        (tmp_path / "bundle" / "deep" / f).write_text("x")
    (tmp_path / "calibre").mkdir()
    for f in ("Title - Author.azw3", "Title - Author retail.pdf"):
        (tmp_path / "calibre" / f).write_text("x")
    for f in ("one.pdf", "two.pdf", "notes.txt"):                 # different books side by side
        (tmp_path / f).write_text("x")
    found = dict(knowledge.find(tmp_path))
    assert found["bundle/deep/deep-learning.epub"] == ["bundle/deep/deep-learning.mobi", "bundle/deep/deep-learning.pdf"]
    assert found["calibre/Title - Author retail.pdf"] == ["calibre/Title - Author.azw3"]   # read from a copy it can convert
    assert {"one.pdf", "two.pdf", "notes.txt"} <= set(found)


def test_scan_converts_skips_and_finds_duplicates(tmp_path):
    lib = knowledge.Library(tmp_path)
    folder = tmp_path / "knowledge" / "documents"
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
    ps = knowledge_skills.parts(text, 700)
    assert all(len(p.split()) <= 700 for p in ps) and sum(len(p.split()) for p in ps) == len(text.split())
    assert [t for t, _ in knowledge_skills.chapters(text)] == ["Front matter"] + [f"Chapter {i}" for i in range(5)]
    assert knowledge_skills.chapters("no headings") == [("Book", "no headings")]
    assert knowledge_skills.part_words(8192) == int((8192 - 3000) / 1.4) and knowledge_skills.part_words(0) == 12000


def test_make_skill_with_a_model():
    asked = []

    def chat(prompt):
        asked.append(prompt)
        if "SKILL.md" in prompt:
            return "<think>hmm</think>```markdown\n---\nname: x\ndescription: Use when \"baking\" bread.\n---\n\n# Bread\n\n## How to do it\n1. Knead\n```"
        return "NOTHING" if "PART 2 " in prompt else "- knead for 10 minutes"
    text = "\n\n".join(WORDS for _ in range(10))
    md = knowledge_skills.make("Bread Book - Ann", text, chat, ctx=4500)
    assert md.startswith("---\nname: bread-book-ann\ndescription: \"Use when 'baking' bread.\"\n---\n\n# Bread")
    assert sum("--- PART" in p for p in asked) == len(asked) - 1
    with pytest.raises(ValueError):
        knowledge_skills.make("x", text, lambda p: "NOTHING", ctx=0)


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
        if svc.knowledge_task and not svc.knowledge_task.running:
            return svc.knowledge_task
        time.sleep(0.01)
    raise AssertionError("book work did not finish")


def test_send_books_to_open_webui(stack):
    app, client, backend, data = stack
    svc = app.state.svc
    epub(data / "knowledge" / "documents" / "Cooking" / "bread.epub", title="Bread")
    epub(data / "knowledge" / "documents" / "Cooking" / "broken.epub", title="Broken", chapters=[("X", "other " + WORDS)])
    epub(data / "knowledge" / "documents" / "Code" / "py.epub", title="Python", chapters=[("Y", "python " + WORDS)])
    r = client.post("/api/v1/knowledge/send", json={"instance": "webui"})
    assert r.status_code == 409 and "Start" in r.json()["error"]
    start(app, client, "webui")
    assert "No documents ready" in client.post("/api/v1/knowledge/send", json={"instance": "webui"}).json()["error"]
    assert client.post("/api/v1/knowledge/scan").status_code == 202
    assert finish(svc).result.startswith("Found the documents: 3 ready")
    assert client.post("/api/v1/knowledge/send", json={"instance": "webui", "name": "Library", "split": True}).status_code == 202
    t = finish(svc)
    assert not t.error, t.error
    assert sorted(FakeOpenWebUI.made) == ["Library - Code", "Library - Cooking"]
    assert sorted(FakeOpenWebUI.uploads) == ["Bread - Ann Author.md", "Python - Ann Author.md"]
    assert "1 failed" in t.result and "Broken" in t.errors[0]
    # again: only what isn't there yet
    client.post("/api/v1/knowledge/send", json={"instance": "webui", "name": "Library", "split": True})
    assert finish(svc).result.startswith("Sent 0 documents") and len(FakeOpenWebUI.uploads) == 2
    listed = client.get("/api/v1/knowledge").json()
    assert listed["sends"][0]["name"] == "Library" and listed["webuis"][0]["running"]
    page = browser(app).get("/knowledge").text
    assert "Knowledge for Open WebUI" in page and "In webui: Library - Cooking" in page


def test_send_asks_before_changing_open_webuis_embeddings(stack, monkeypatch):
    app, client, backend, data = stack
    svc = app.state.svc
    start(app, client, "webui")
    epub(data / "knowledge" / "documents" / "bread.epub")
    svc.scan_documents()
    finish(svc)
    from gpushepherd import instance_templates as tpl
    m = svc.machine()
    m.servers["llm"]["embed"] = "embed"
    monkeypatch.setattr(svc, "machine", lambda: m)
    client.post("/api/v1/knowledge/send", json={"instance": "webui"})
    assert "Choose which to use" in finish(svc).error and not FakeOpenWebUI.uploads
    client.post("/api/v1/knowledge/send", json={"instance": "webui", "embeddings": "server"})
    assert not finish(svc).error
    assert FakeOpenWebUI.switched == [(tpl._server_url("llm", m), "embed")]


def test_upload_folder_and_cli(stack, capsys):
    app, client, backend, data = stack
    svc = app.state.svc
    path = epub(data / "x" / "up.epub")
    with open(path, "rb") as f:
        r = client.post("/api/v1/knowledge/upload", files=[("file", ("up.epub", f)), ("file", ("evil.exe", b"x"))])
    assert r.status_code == 202 and "left out evil.exe" in r.json()["message"]
    finish(svc)
    assert (data / "knowledge" / "documents" / "up.epub").exists() and not (data / "knowledge" / "documents" / "evil.exe").exists()
    assert "can't see" in client.put("/api/v1/knowledge/folder", json={"path": "/nope/books"}).json()["error"]
    assert client.put("/api/v1/knowledge/folder", json={"path": str(data / "x")}).status_code == 200
    assert client.get("/api/v1/knowledge").json()["folder"] == str(data / "x")


def test_bulk_upload_keeps_folders_unpacks_zips_and_skips(stack, monkeypatch):
    app, client, backend, data = stack
    svc = app.state.svc
    book = epub(data / "x" / "a.epub").read_bytes()
    r = client.put("/api/v1/knowledge/files", params={"path": "Bundle One/Cooking/a.epub"}, content=book)
    assert r.status_code == 200 and r.json()["added"] == ["Bundle One/Cooking/a.epub"]
    zpath = data / "x" / "Bundle Two.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("Python/b.epub", epub(data / "x" / "b.epub", title="Python").read_bytes())
        z.writestr("readme.html", "<p>hi</p>")
        z.writestr("Python/readme.txt", "a short readme, not a book")
        z.writestr("../evil.epub", "x")
        z.writestr("__MACOSX/Python/._b.epub", "x")
    r = client.put("/api/v1/knowledge/files", params={"path": "Bundle Two.zip"}, content=zpath.read_bytes())
    assert r.json() == {"added": ["Bundle Two/Python/b.epub"], "left_out": 3}
    folder = data / "knowledge" / "documents"
    assert (folder / "Bundle Two" / "Python" / "b.epub").exists() and not (folder / "Bundle Two.zip").exists()
    assert not (data / "evil.epub").exists() and not list(folder.rglob("*.part"))
    assert client.get("/api/v1/knowledge/files").json()["files"] == {
        "Bundle One/Cooking/a.epub": len(book), "Bundle Two/Python/b.epub": (data / "x" / "b.epub").stat().st_size}
    # refused: outside the folder, not a book, too big, a zip with no books
    for path in ("../up.epub", "/etc/x/../../up.epub", "a/.hidden/b.epub", "run.exe", ""):
        assert client.put("/api/v1/knowledge/files", params={"path": path}, content=b"x").status_code == 400, path
    monkeypatch.setattr(knowledge, "UPLOAD_MB", 0)
    r = client.put("/api/v1/knowledge/files", params={"path": "big.epub"}, content=b"x" * 10)
    assert r.status_code == 413 and not (folder / "big.epub").exists() and not (folder / "big.epub.part").exists()
    monkeypatch.setattr(knowledge, "UPLOAD_MB", 200)
    empty = data / "x" / "none.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("notes.html", "x")
    assert "no documents" in client.put("/api/v1/knowledge/files", params={"path": "none.zip"}, content=empty.read_bytes()).json()["error"]
    err = client.put("/api/v1/knowledge/files", params={"path": "Sub/bad.zip"}, content=b"nope").json()["error"]
    assert err.startswith("bad.zip: can't open") and ".part" not in err
    # the dashboard's route: signed in and from the page only
    b = browser(app)
    assert b.put("/knowledge/files", params={"path": "c.txt"}, content=WORDS.encode()).status_code == 403
    r = b.put("/knowledge/files", params={"path": "Notes/c.txt"}, content=WORDS.encode(), headers={"HX-Request": "true"})
    assert r.json()["added"] == ["Notes/c.txt"]
    assert "Notes/c.txt" in b.get("/knowledge/files").json()["files"]
    assert 'id="book-drop"' in b.get("/knowledge").text
    assert client.post("/api/v1/knowledge/scan").status_code == 202
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
    cli.upload_documents(Via(), [lib])
    assert (data / "knowledge" / "documents" / "Library" / "Bread" / "bread.epub").exists()
    assert not (data / "knowledge" / "documents" / "Library" / ".cache").exists()
    assert "Added 1 document," in capsys.readouterr().out
    finish(app.state.svc)
    cli.upload_documents(Via(), [lib])
    assert "Added 0 documents, skipped 1 already there." in capsys.readouterr().out


def test_skill_made_checked_and_installed(stack, monkeypatch):
    app, client, backend, data = stack
    svc = app.state.svc
    epub(data / "knowledge" / "documents" / "bread.epub", title="Bread")
    svc.scan_documents()
    finish(svc)
    key = svc.library.books()[0].key
    assert "Start" in client.post(f"/api/v1/knowledge/{key}/skill", json={"server": "llm"}).json()["error"]
    start(app, client, "llm")
    monkeypatch.setattr(svc, "_chat", lambda server, m: lambda p: (
        "---\nname: x\ndescription: Use when baking.\n---\n\n# Bread\n1. Knead" if "SKILL.md" in p else "- knead"))
    assert client.post(f"/api/v1/knowledge/{key}/skill", json={"server": "llm"}).status_code == 202
    t = finish(svc)
    assert not t.error, t.error
    slug = "bread-ann-author"
    draft = data / "knowledge" / "skills" / slug
    assert (draft / "SKILL.md").read_text().startswith(f"---\nname: {slug}\n")
    assert sorted(p.name for p in (draft / "references").iterdir()) == ["01-front-matter.md", "02-one.md", "03-two.md"]
    assert "1. Knead" in client.get(f"/api/v1/skills/{slug}").json()["text"]
    bad = client.put(f"/api/v1/skills/{slug}", json={"text": "# no frontmatter"})
    assert bad.status_code == 400
    assert client.put(f"/api/v1/skills/{slug}", json={"text": f"---\nname: {slug}\ndescription: Use when.\n---\n# Edited"}).status_code == 200
    assert "isn't Hermes" in client.post(f"/api/v1/skills/{slug}/install", json={"instance": "webui"}).json()["error"]
    assert client.post(f"/api/v1/skills/{slug}/install", json={"instance": "hermes"}).status_code == 200
    installed = data / "appdata" / "hermes" / "skills" / "knowledge" / slug / "SKILL.md"
    assert installed.read_text().endswith("# Edited\n")
    assert client.get("/api/v1/knowledge").json()["skills"][0]["installed"].keys() == {"hermes"}
    page = browser(app).get("/knowledge").text
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


# --------------------------------------------------------------------------- reading what EPUBs in the wild get wrong

def odd_epub(path: Path, opf: str | None, files: dict[str, str], container: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        if container:
            z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                       '<rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>')
        if opf is not None:
            z.writestr("OPS/package.opf", opf)
        for name, body in files.items():
            z.writestr(name, body)
    return path


def page_html(text):
    return f'<?xml version="1.0" encoding="utf-8"?><html><body><p>{text}</p><table><tr><td>a</td><td>b</td></tr></table></body></html>'


def test_epubs_with_escaped_names_broken_packages_and_no_reading_order(tmp_path):
    # hrefs escaped (%20) and in another case, an &nbsp; in the package, no namespace, no spine
    opf = ('<package><metadata><title>Odd&nbsp;Book</title><creator>Bo</creator></metadata><manifest>'
           '<item id="a" href="Text/Chapter%201.xhtml" media-type="application/xhtml+xml"/>'
           '<item id="b" href="text/chapter 2.XHTML" media-type="application/xhtml+xml"/></manifest></package>')
    p = odd_epub(tmp_path / "odd.epub", opf, {"OPS/Text/Chapter 1.xhtml": page_html("one " + WORDS),
                                              "OPS/Text/Chapter 2.xhtml": page_html("two " + WORDS)})
    title, author, text = knowledge.read(p)
    assert (title, author) == ("Odd\xa0Book", "Bo") and text.index("one") < text.index("two") and "a b" in text
    # no container and a package that can't be parsed: every page, by name
    p = odd_epub(tmp_path / "bare.epub", "<package><<<", {"OPS/x/1.html": page_html("first " + WORDS),
                                                         "OPS/x/2.html": page_html("second " + WORDS)}, container=False)
    assert knowledge.read(p)[2].index("first") < knowledge.read(p)[2].index("second")
    # a rights.xml left behind by a shop, with readable text: not DRM
    p = odd_epub(tmp_path / "rights.epub", None, {"META-INF/rights.xml": "<rights/>", "OPS/a.xhtml": page_html(WORDS)})
    assert "word5" in knowledge.read(p)[2]
    # pictures only
    p = odd_epub(tmp_path / "pics.epub", None, {"OPS/a.xhtml": "<html><body><img src='a.jpg'/></body></html>"})
    with pytest.raises(knowledge.BookError, match="pictures"):
        knowledge.read(p)
    # not an EPUB at all (a broken download)
    (tmp_path / "broken.epub").write_bytes(b"\x00" * 100)
    with pytest.raises(knowledge.BookError, match="damaged"):
        knowledge.read(tmp_path / "broken.epub")


def test_docx_and_short_notes(tmp_path):
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    doc = tmp_path / "manual.docx"
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{w}"><w:body>'
                   f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Setup</w:t></w:r></w:p>'
                   f'<w:p><w:r><w:t>Plug it in and </w:t></w:r><w:r><w:t>turn it on. {WORDS[:200]}</w:t></w:r></w:p>'
                   '</w:body></w:document>')
        z.writestr("docProps/core.xml", '<cp:coreProperties xmlns:cp="x" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:title>The Manual</dc:title><dc:creator>Acme</dc:creator></cp:coreProperties>')
    title, author, text = knowledge.read(doc)
    assert (title, author) == ("The Manual", "Acme") and text.startswith("# Setup\n\nPlug it in and turn it on.")
    note = tmp_path / "note.md"
    note.write_text("Thirty words " * 10)
    assert knowledge.read(note)[2]
    (tmp_path / "README.txt").write_text("short readme")
    (tmp_path / "LICENSE.md").write_text("short licence")
    assert {p for p, _ in knowledge.find(tmp_path)} == {"manual.docx", "note.md"}


def test_unreadable_documents_are_tried_again_when_the_reader_gets_better(tmp_path, monkeypatch):
    lib = knowledge.Library(tmp_path)
    folder = lib.default_folder
    folder.mkdir(parents=True)
    epub(folder / "a.epub")
    monkeypatch.setattr(knowledge, "read", lambda p: (_ for _ in ()).throw(knowledge.BookError("can't yet")))
    assert lib.scan() == {"failed": 1}
    monkeypatch.undo()
    assert lib.scan() == {"failed": 1}                       # same reader: not read again
    monkeypatch.setattr(knowledge, "READER", knowledge.READER + 1)
    assert lib.scan() == {"ready": 1}


def test_an_older_install_is_moved_into_the_knowledge_folder(tmp_path):
    epub(tmp_path / "books" / "Bundle" / "a.epub")
    (tmp_path / "book-text").mkdir()
    (tmp_path / "book-text" / "k.md").write_text("x")
    (tmp_path / "book-skills" / "s").mkdir(parents=True)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "books.yaml").write_text(yaml.safe_dump({"folder": "", "books": {}, "skills": {"s": {"name": "S"}}}))
    lib = knowledge.Library(tmp_path)
    home = tmp_path / "knowledge"
    assert (home / "documents" / "Bundle" / "a.epub").exists() and not (tmp_path / "books").exists()
    assert (home / "text" / "k.md").exists() and (home / "skills" / "s").is_dir()
    assert not (tmp_path / "state" / "books.yaml").exists() and lib.load()["skills"] == {"s": {"name": "S"}}
    assert lib.folder() == home / "documents" and len(lib.moved) == 4
    assert knowledge.Library(tmp_path).moved == []          # once
    # a folder of the user's own is left where it is
    other = tmp_path / "other"
    (other / "books").mkdir(parents=True)
    (other / "state").mkdir()
    (other / "state" / "books.yaml").write_text(yaml.safe_dump({"folder": "/srv/mine", "books": {}}))
    lib = knowledge.Library(other)
    assert (other / "books").is_dir() and str(lib.folder()) == "/srv/mine"


def test_storage_reasons_clear_and_old_names(stack):
    app, client, backend, data = stack
    svc = app.state.svc
    folder = data / "knowledge" / "documents"
    epub(folder / "Cooking" / "bread.epub", title="Bread")
    (folder / "Cooking" / "bad.epub").write_bytes(b"nope")
    (folder / "Cooking" / "worse.epub").write_bytes(b"nope")
    start(app, client, "webui")
    client.post("/api/v1/knowledge/scan")
    finish(svc)
    k = client.get("/api/v1/books").json()                   # the old name still answers
    assert k["counts"] == {"ready": 1, "failed": 2} and len(k["reasons"]["failed"]) == 1
    assert k["reasons"]["failed"][0][1] == 2 and "damaged" in k["reasons"]["failed"][0][0]
    docs = {x["what"]: x for x in k["storage"]}
    assert docs["Documents"]["path"] == str(folder) and docs["Documents"]["files"] == 3
    assert docs["Converted text"]["files"] == 1 and k["own_folder"]
    client.post("/api/v1/knowledge/send", json={"instance": "webui"})
    assert not finish(svc).error
    key = next(x["key"] for x in k["documents"] if x["status"] == "ready")
    # a skill, installed in Hermes
    slug = "bread-ann-author"
    (data / "knowledge" / "skills" / slug).mkdir(parents=True)
    (data / "knowledge" / "skills" / slug / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n")
    svc.library.update(lambda d: d.setdefault("skills", {}).__setitem__(slug, {"book": key, "name": "Bread"}))
    assert client.post(f"/api/v1/skills/{slug}/install", json={"instance": "hermes"}).status_code == 200
    installed = data / "appdata" / "hermes" / "skills" / "knowledge" / slug
    old_copy = data / "appdata" / "hermes" / "skills" / "books" / slug
    old_copy.mkdir(parents=True)
    assert installed.is_dir()

    removed, deleted, gone = [], [], []
    FakeOpenWebUI.knowledge_bases = lambda self: dict(self.made)
    FakeOpenWebUI.remove = lambda self, kid, fid: removed.append((kid, fid))
    FakeOpenWebUI.delete_file = lambda self, fid: deleted.append(fid)
    FakeOpenWebUI.knowledge_empty = lambda self, kid: True
    FakeOpenWebUI.delete_knowledge = lambda self, kid: gone.append(kid)
    page = browser(app).get("/knowledge?show=failed").text
    assert page.count("bad.epub") + page.count("worse.epub") >= 2 and "bread.epub" not in page
    assert browser(app).get("/books", follow_redirects=False).headers["location"] == "/knowledge"
    assert client.post("/api/v1/knowledge/clear", json={"what": ["nothing"]}).status_code == 400
    assert client.post("/api/v1/knowledge/clear").status_code == 202
    t = finish(svc)
    assert not t.error, t.error
    assert removed == [("kb0", "f1")] and deleted == ["f1"] and gone == ["kb0"]
    assert not installed.exists() and not old_copy.exists() and not (data / "knowledge" / "skills").exists()
    assert folder.is_dir() and not any(folder.iterdir())
    k = client.get("/api/v1/knowledge").json()
    assert k["documents"] == [] and k["skills"] == [] and k["sends"] == []
    assert not (data / "knowledge" / "text").exists()


def test_clearing_a_folder_of_your_own_keeps_other_files(stack):
    app, client, backend, data = stack
    svc = app.state.svc
    mine = data / "mine"
    epub(mine / "a.epub")
    (mine / "photo.jpg").write_bytes(b"x")
    client.put("/api/v1/knowledge/folder", json={"path": str(mine)})
    assert client.post("/api/v1/knowledge/clear", json={"what": "documents,converted"}).status_code == 202
    assert "deleted 1 file" in finish(svc).result
    assert (mine / "photo.jpg").exists() and not (mine / "a.epub").exists()


def test_upload_in_pieces_resumes_and_puts_the_file_in_place(stack):
    app, client, backend, data = stack
    book = epub(data / "x" / "long.epub").read_bytes()
    a, b, total = book[:1000], book[1000:], len(book)
    path = "A very long folder name/" + "x" * 150 + "/long.epub"
    for c in (client, browser(app)):
        route = "/api/v1/knowledge/files" if c is client else "/knowledge/files"
        h = {} if c is client else {"HX-Request": "true"}
        r = c.put(route, params={"path": path, "offset": 0, "total": total}, content=a, headers=h)
        assert r.json() == {"received": 1000}
        r = c.put(route, params={"path": path, "offset": 5, "total": total}, content=b, headers=h)
        assert r.status_code == 409 and r.json()["have"] == 1000             # out of order: says where to go on
        r = c.put(route, params={"path": path, "offset": 1000, "total": total}, content=b + b"extra", headers=h)
        assert r.status_code == 400 and r.json().get("have") is None
        assert (data / "knowledge" / "documents" / (path + ".part")).stat().st_size == 1000   # the bad piece was cut off
        r = c.put(route, params={"path": path, "offset": 1000, "total": total}, content=b, headers=h)
        assert r.json()["added"] == [path]
        assert (data / "knowledge" / "documents" / path).read_bytes() == book
        assert not (data / "knowledge" / "documents" / (path + ".part")).exists()
    assert client.put("/api/v1/knowledge/files", params={"path": "e.txt", "offset": "x", "total": 3}).status_code == 400


def blank_pdf(path: Path, pages=3) -> Path:
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(200, 200)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        w.write(f)
    return path


def test_pdf_text_pypdf_misses_is_read_with_pdfium(tmp_path, monkeypatch):
    p = pdf(tmp_path / "a.pdf", WORDS)
    monkeypatch.setattr(knowledge, "_pypdf_text", lambda path: ("T", "A", "", 3))
    assert "word5" in knowledge.read(p)[2]


def test_scanned_pdfs_are_read_with_ocr(tmp_path, monkeypatch):
    lib = knowledge.Library(tmp_path)
    blank_pdf(lib.default_folder / "scan.pdf")
    epub(lib.default_folder / "ok.epub")
    assert lib.scan() == {"ready": 1, "scanned": 1}
    assert "OCR" in {b.path: b for b in lib.books()}["scan.pdf"].reason
    tess = tmp_path / "bin" / "tesseract"
    tess.parent.mkdir()
    tess.write_text("#!/bin/sh\nhead -c 2 \"$1\" | grep -q P5 && echo 'scanned page text " + " ".join(f"w{i}" for i in range(30)) + "'\n")
    tess.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tess.parent}:{__import__('os').environ['PATH']}")
    seen = []
    done, errors = lib.ocr(lambda d, t, text: seen.append(text))
    assert (done, errors) == (1, []) and any("page" in s for s in seen)
    b = {b.path: b for b in lib.books()}["scan.pdf"]
    assert b.status == "ready" and b.ocr and lib.text(b.key).count("scanned page text") == 3
    assert lib.scan() == {"ready": 2}                         # kept at the next scan


def test_delete_one_document(stack):
    app, client, backend, data = stack
    svc = app.state.svc
    folder = data / "knowledge" / "documents"
    epub(folder / "Cooking" / "bread.epub", title="Bread")
    (folder / "Cooking" / "bread.mobi").write_text("x")
    epub(folder / "keep.epub", title="Keep", chapters=[("K", "keep " + WORDS)])
    blank_pdf(folder / 'Say "cheese".pdf')
    client.post("/api/v1/knowledge/scan")
    finish(svc)
    page = browser(app).get("/knowledge").text
    assert "Needs OCR" in page and "Read 1 with OCR" in page and page.count(">Delete</button>") == 3
    assert 'Delete Say &#34;cheese&#34;?' in page or "Delete Say &quot;cheese&quot;?" in page
    key = next(b.key for b in svc.library.books() if b.path == "Cooking/bread.epub")
    svc.library.update(lambda d: d["books"][key].sent.update({"webui/Documents": "f9"}))
    start(app, client, "webui")
    removed, deleted = [], []
    FakeOpenWebUI.knowledge_bases = lambda self: {"Documents": "kb7"}
    FakeOpenWebUI.remove = lambda self, kid, fid: removed.append((kid, fid))
    FakeOpenWebUI.delete_file = lambda self, fid: deleted.append(fid)
    r = browser(app).post(f"/knowledge/{key}/delete", headers={"HX-Request": "true"})
    assert r.status_code == 204 and "Deleted" in r.headers["HX-Redirect"]
    assert not (folder / "Cooking").exists() and (folder / "keep.epub").exists()
    assert removed == [("kb7", "f9")] and deleted == ["f9"] and not svc.library.text_path(key).exists()
    assert sorted(b.path for b in svc.library.books()) == ["Say \"cheese\".pdf", "keep.epub"]
    assert client.delete(f"/api/v1/knowledge/{key}").status_code == 404
