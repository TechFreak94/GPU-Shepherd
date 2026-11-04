"""Books: ebooks turned into plain text that chat apps can search (and agents can learn from).

A books folder (<data>/books unless another one GPU Shepherd can see is chosen) holds the ebooks,
in any folder layout. A scan finds them and converts each one to Markdown:

    EPUB, PDF, TXT, Markdown    converted here (PDF with pypdf)
    AZW3, MOBI, PRC             only counted as other copies of a book: converting them needs Calibre

A book often comes in several formats: copies in the same folder with the same name. Only the best
copy is converted: EPUB > AZW3 > MOBI > PRC > PDF > Markdown > TXT. A book whose text matches one
already found (the same book from two bundles, under another name) is a duplicate and skipped. Books that
can't be read (DRM, a scanned PDF with no text) are listed with the reason.

    <data>/state/books.yaml     the books found and what happened to them (the index)
    <data>/book-text/<key>.md   each book's text

Pure file work: no network and no containers here. Sending books to an app is in service.py
with connectors/openwebui.py; skills are made in book_skills.py.
"""
from __future__ import annotations

import hashlib
import html
import os
import posixpath
import re
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import yaml

from .config import load_yaml

# the lower, the better a copy to read the book from
RANK = {".epub": 0, ".azw3": 1, ".mobi": 2, ".prc": 3, ".pdf": 4, ".md": 5, ".markdown": 5, ".txt": 6}
CONVERTS = {".epub", ".pdf", ".md", ".markdown", ".txt"}
MIN_WORDS = 200              # less text than this: nothing to search (a scanned PDF, an empty file)
SAMPLE_WORDS = 400           # words compared to find the same book under another name
UPLOAD_MB = 4096             # the biggest book the dashboard takes as an upload (streamed to disk, so only free space matters)
ZIP_MB = 65536               # the biggest zip of books it takes (unpacked into the books folder)


class BookError(Exception):
    """Why a book can't be read."""


@dataclass
class Book:
    key: str                     # stable id, from where the book's best copy is
    path: str                    # its best copy, relative to the books folder
    group: str = ""              # its first folder inside the books folder ("" at the top)
    title: str = ""
    author: str = ""
    status: str = "new"          # ready | failed | skipped | duplicate
    reason: str = ""
    words: int = 0
    other_copies: list[str] = field(default_factory=list)   # other formats of the same book
    same_as: str = ""            # duplicate: the key of the book it repeats
    fingerprint: str = ""
    stamp: str = ""              # size and time of the copy the text came from
    sent: dict[str, str] = field(default_factory=dict)      # "<app>/<knowledge>" -> the app's file id

    @property
    def name(self) -> str:
        """What the book is called in apps: its title and author."""
        title = self.title or Path(self.path).stem
        return f"{title} - {self.author}" if self.author else title

    @property
    def file_name(self) -> str:
        """A file name for the book's text, readable in an app's list of sources."""
        safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", self.name)
        return re.sub(r"\s+", " ", safe).strip()[:150] + ".md"


# --------------------------------------------------------------------------- finding books

def _stem(name: str) -> str:
    """A file name without its format and the bits that differ between copies of one book."""
    s = Path(name).stem.lower()
    s = re.sub(r"[\s_.\-]+", " ", s)
    return re.sub(r"\b(epub|pdf|mobi|azw3|retail|ebook|v\d+)\b", "", s).strip()


def find(folder: Path) -> list[tuple[str, list[str]]]:
    """The books in a folder: for each, its best copy and the other copies (paths relative to
    the folder). Copies of one book are files in the same folder with the same name (apart from
    their format and words like "retail"); the same book under other names is found by its text."""
    by_dir: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_duplicates")))
        rel = os.path.relpath(root, folder)
        for f in sorted(files):
            if Path(f).suffix.lower() in RANK and not f.startswith("."):
                by_dir.setdefault("" if rel == "." else rel, []).append(f)
    out = []
    for d, files in by_dir.items():
        named: dict[str, list[str]] = {}
        for f in files:
            named.setdefault(_stem(f) or f, []).append(f)
        for g in named.values():
            g = sorted(g, key=lambda f: (RANK[Path(f).suffix.lower()], f))
            best = next((f for f in g if Path(f).suffix.lower() in CONVERTS), g[0])
            paths = [posixpath.join(d, f) if d else f for f in g]
            best = posixpath.join(d, best) if d else best
            out.append((best, [p for p in paths if p != best]))
    return sorted(out)


def safe_rel(path: str) -> str:
    """An uploaded file's place in the books folder, from the path the browser gives (folders kept),
    or BookError if it would land outside the folder or isn't a book or a zip of books."""
    parts = [p.strip() for p in str(path).replace("\\", "/").split("/")]
    parts = [p for p in parts if p and p != "."]
    if not parts or ".." in parts or any(p.startswith(".") or ":" in p or any(ord(c) < 32 for c in p) for p in parts):
        raise BookError(f"{path or '?'}: not a usable file name")
    if len(parts) > 12 or len("/".join(parts)) > 400:
        raise BookError(f"{path}: folders too deep or name too long")
    if Path(parts[-1]).suffix.lower() not in RANK and Path(parts[-1]).suffix.lower() != ".zip":
        raise BookError(f"{parts[-1]}: not a book (EPUB, PDF, TXT, Markdown, AZW3, MOBI, PRC) or a zip of books")
    return "/".join(parts)


def files(folder: Path) -> dict[str, int]:
    """The book files in a folder and their sizes, by path relative to it (to skip uploading them again)."""
    out = {}
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel = os.path.relpath(root, folder)
        for f in names:
            if Path(f).suffix.lower() in RANK and not f.startswith("."):
                try:
                    out[posixpath.join(rel, f) if rel != "." else f] = (Path(root) / f).stat().st_size
                except OSError:
                    pass
    return out


def unpack(zip_path: Path, into: Path) -> tuple[list[str], int]:
    """Take the books out of a zip into a folder, keeping the zip's folders. Other files are left
    out. Returns the books' paths relative to `into` and how many other files were left out."""
    saved, other = [], 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            if sum(m.file_size for m in members) > ZIP_MB << 20:
                raise BookError(f"{zip_path.name} unpacks to more than {ZIP_MB // 1024} GB")
            free = shutil.disk_usage(into if into.exists() else into.parent).free
            if sum(m.file_size for m in members) > free:
                raise BookError(f"{zip_path.name}: not enough free space to unpack it")
            for m in members:
                if m.filename.startswith(("__MACOSX/", "/")):
                    continue
                try:
                    rel = safe_rel(m.filename)
                except BookError:
                    other += 1
                    continue
                if rel.lower().endswith(".zip"):
                    other += 1            # a zip in a zip: not opened
                    continue
                dest = into / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                with z.open(m) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.replace(tmp, dest)
                saved.append(rel)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError) as err:
        raise BookError(f"{zip_path.name}: can't open the zip ({err})") from err
    return saved, other


def book_key(rel: str) -> str:
    return hashlib.sha1(rel.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- reading books

class _Text(HTMLParser):
    """HTML to Markdown-ish text: headings, paragraphs and list items; scripts and styles left out."""
    BLOCK = {"p", "div", "section", "article", "blockquote", "tr", "table", "ul", "ol", "dl", "dd", "dt",
             "figure", "figcaption", "pre", "hr", "body"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(self.skip - 1, 0)
        elif tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n[ \t]+", "\n", t)
        t = re.sub(r"\n#+ \n", "\n", t)            # headings with nothing in them
        return re.sub(r"\n{3,}", "\n\n", t).strip()


def html_text(markup: str) -> str:
    p = _Text()
    p.feed(markup)
    p.close()
    return p.text()


_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container", "opf": "http://www.idpf.org/2007/opf",
       "dc": "http://purl.org/dc/elements/1.1/", "enc": "http://www.w3.org/2001/04/xmlenc#"}
_FONT_OBFUSCATION = ("http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC")


def read_epub(path: Path) -> tuple[str, str, str]:
    """An EPUB's title, author and text, its chapters in reading order."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as err:
        raise BookError(f"not a readable EPUB ({err})") from err
    with z:
        names = set(z.namelist())
        if "META-INF/rights.xml" in names:
            raise BookError("protected by DRM")
        try:
            container = ElementTree.fromstring(z.read("META-INF/container.xml"))
            opf_path = container.find(".//c:rootfile", _NS).get("full-path")
            opf = ElementTree.fromstring(z.read(opf_path))
        except (KeyError, AttributeError, ElementTree.ParseError) as err:
            raise BookError("not a readable EPUB (no package file)") from err
        base = posixpath.dirname(opf_path)
        items = {i.get("id"): posixpath.normpath(posixpath.join(base, html.unescape(i.get("href", ""))))
                 for i in opf.iterfind(".//opf:manifest/opf:item", _NS)}
        spine = [items[r.get("idref")] for r in opf.iterfind(".//opf:spine/opf:itemref", _NS) if r.get("idref") in items]
        if "META-INF/encryption.xml" in names:            # fonts may be obfuscated; text must not be encrypted
            enc = ElementTree.fromstring(z.read("META-INF/encryption.xml"))
            for data in enc.iterfind(".//enc:EncryptedData", _NS):
                method = data.find("enc:EncryptionMethod", _NS)
                ref = data.find(".//enc:CipherReference", _NS)
                if ref is not None and (method is None or method.get("Algorithm") not in _FONT_OBFUSCATION) \
                        and posixpath.normpath(ref.get("URI", "")) in spine:
                    raise BookError("protected by DRM")
        title = (opf.findtext(".//dc:title", "", _NS) or "").strip()
        author = (opf.findtext(".//dc:creator", "", _NS) or "").strip()
        chapters = []
        for name in spine:
            try:
                raw = z.read(name).decode("utf-8", "replace")
            except KeyError:
                continue
            t = html_text(raw)
            if t:
                chapters.append(t)
    return title, author, "\n\n".join(chapters)


def read_pdf(path: Path) -> tuple[str, str, str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as err:            # pragma: no cover - a dependency
        raise BookError("reading PDFs needs the pypdf package") from err
    try:
        r = PdfReader(str(path))
        if r.is_encrypted and not r.decrypt(""):
            raise BookError("protected by a password or DRM")
        meta = r.metadata or {}
        pages = [(p.extract_text() or "").strip() for p in r.pages]
    except BookError:
        raise
    except (PdfReadError, OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        raise BookError(f"not a readable PDF ({err.__class__.__name__})") from err
    text = "\n\n".join(re.sub(r"[ \t]+", " ", p) for p in pages if p)
    if len(text.split()) < MIN_WORDS and len(pages) > 2:
        raise BookError("no text in it (a scanned PDF needs OCR first)")
    return str(meta.get("/Title") or "").strip(), str(meta.get("/Author") or "").strip(), text


def read(path: Path) -> tuple[str, str, str]:
    """A book's title, author and text (Markdown). Raises BookError when it can't be read."""
    ext = path.suffix.lower()
    if ext == ".epub":
        title, author, text = read_epub(path)
    elif ext == ".pdf":
        title, author, text = read_pdf(path)
    elif ext in (".md", ".markdown", ".txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        title, author = "", ""
    else:
        raise BookError(f"{ext[1:].upper()} needs Calibre to convert; add an EPUB or PDF copy instead")
    if len(text.split()) < MIN_WORDS:
        raise BookError("almost no text in it")
    return title, author, text


def fingerprint(text: str) -> str:
    """Words from the middle of a book: the same for two copies of it, whatever their names and
    front matter."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    start = max(len(words) // 2 - SAMPLE_WORDS // 2, 0)
    return hashlib.sha1(" ".join(words[start:start + SAMPLE_WORDS]).encode()).hexdigest()[:16]


def markdown(b: Book, text: str) -> str:
    head = f"# {b.title or Path(b.path).stem}\n\n" + (f"By {b.author}\n\n" if b.author else "")
    return head + text.strip() + "\n"


# --------------------------------------------------------------------------- the index

class Library:
    """The books folder, the books found in it and their text."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.index_file = self.root / "state" / "books.yaml"
        self.text_dir = self.root / "book-text"
        self._lock = threading.RLock()

    # the index: {folder: str, books: {key: Book}, sends: [...], skills: {...}}
    def load(self) -> dict:
        with self._lock:
            try:
                data = load_yaml(self.index_file.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                data = {}
            known = set(Book.__dataclass_fields__)
            data["books"] = {k: Book(**{f: v for f, v in b.items() if f in known})
                             for k, b in (data.get("books") or {}).items() if isinstance(b, dict)}
            data.setdefault("folder", "")
            return data

    def save(self, data: dict) -> None:
        with self._lock:
            out = dict(data) | {"books": {k: asdict(b) for k, b in data["books"].items()}}
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.index_file.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
            os.replace(tmp, self.index_file)

    def update(self, fn) -> dict:
        """Load, change with fn(data) and save, as one step."""
        with self._lock:
            data = self.load()
            fn(data)
            self.save(data)
            return data

    def folder(self, data: dict | None = None) -> Path:
        data = data if data is not None else self.load()
        return Path(data.get("folder") or self.root / "books")

    def books(self) -> list[Book]:
        return sorted(self.load()["books"].values(), key=lambda b: (b.group.lower(), b.name.lower()))

    def text_path(self, key: str) -> Path:
        return self.text_dir / f"{key}.md"

    def text(self, key: str) -> str:
        return self.text_path(key).read_text(encoding="utf-8")

    # ------------------------------------------------------------------ scanning
    def scan(self, progress=lambda done, total, name: None, stop=lambda: False) -> dict:
        """Find the books in the folder and convert the new and changed ones. Returns counts."""
        folder = self.folder()
        if not folder.is_dir():
            raise BookError(f"GPU Shepherd can't see the books folder {folder}")
        found = find(folder)
        old = self.load()["books"]
        books: dict[str, Book] = {}
        self.text_dir.mkdir(parents=True, exist_ok=True)
        for i, (rel, others) in enumerate(found):
            if stop():
                break
            key = book_key(rel)
            src = folder / rel
            try:
                st = src.stat()
            except OSError:
                continue
            stamp = f"{st.st_size}:{int(st.st_mtime)}"
            prev = old.get(key)
            b = Book(key=key, path=rel, group=rel.split("/", 1)[0] if "/" in rel else "", other_copies=others,
                     stamp=stamp)
            if prev and prev.stamp == stamp and (prev.status != "ready" or self.text_path(key).exists()):
                b.title, b.author, b.status, b.reason = prev.title, prev.author, prev.status, prev.reason
                b.words, b.fingerprint, b.sent = prev.words, prev.fingerprint, prev.sent
            else:
                progress(i, len(found), rel)
                try:
                    b.title, b.author, text = read(src)
                    b.status, b.words, b.fingerprint = "ready", len(text.split()), fingerprint(text)
                    self.text_path(key).write_text(markdown(b, text), encoding="utf-8")
                except BookError as err:
                    b.status = "skipped" if src.suffix.lower() not in CONVERTS else "failed"
                    b.reason = str(err)
                except Exception as err:  # noqa: BLE001 - one bad file must not stop the rest
                    b.status, b.reason = "failed", f"couldn't read it ({err.__class__.__name__}: {err})"[:200]
            if b.status == "duplicate":
                b.status, b.same_as = "ready", ""        # worked out again below
            books[key] = b
        stopped = stop()
        if stopped:                                      # keep what wasn't looked at this time
            for k, b in old.items():
                books.setdefault(k, b)
        # the same text under another name: keep the copy in the better format (then the first found)
        seen: dict[str, Book] = {}
        for b in sorted(books.values(), key=lambda b: (RANK.get(Path(b.path).suffix.lower(), 9), b.path)):
            if b.status != "ready" or not b.fingerprint:
                continue
            if b.fingerprint in seen:
                b.status, b.same_as = "duplicate", seen[b.fingerprint].key
                b.reason = f"the same text as {seen[b.fingerprint].path}"
            else:
                seen[b.fingerprint] = b
        for f in self.text_dir.glob("*.md"):            # text of books no longer there
            if f.stem not in books:
                f.unlink(missing_ok=True)
        self.update(lambda d: d.__setitem__("books", books) or d.__setitem__("scanned", time.time()))
        counts: dict[str, int] = {}
        for b in books.values():
            counts[b.status] = counts.get(b.status, 0) + 1
        return counts


# --------------------------------------------------------------------------- background work

class Task:
    """One piece of book work running in the background (finding books, sending them, making a
    skill), for the Books page to show. Only one runs at a time."""

    def __init__(self, title: str):
        self.title = title
        self.text = "Starting…"
        self.done = 0
        self.total = 0
        self.started = time.time()
        self.finished: float | None = None
        self.result = ""
        self.error = ""
        self.errors: list[str] = []
        self.stop = False
        self.stopped = False

    @property
    def running(self) -> bool:
        return self.finished is None

    def progress(self, done: int, total: int, text: str) -> None:
        self.done, self.total, self.text = done, total, text

    def finish(self, result: str, stopped: bool = False) -> None:
        self.result, self.stopped, self.finished = result, stopped, time.time()

    def fail(self, error: str) -> None:
        self.error, self.finished = error, time.time()

    def as_dict(self) -> dict:
        pct = round(100 * self.done / self.total) if self.total else None
        return {"title": self.title, "text": self.text, "done": self.done, "total": self.total, "pct": pct,
                "running": self.running, "result": self.result, "error": self.error, "stopped": self.stopped,
                "errors": self.errors[-20:], "started": self.started, "finished": self.finished}
