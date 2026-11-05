"""Knowledge: documents (ebooks, manuals, notes) turned into plain text that chat apps can search
(and agents can learn from).

Everything lives in one folder, <data>/knowledge:

    documents/          the documents (unless another folder GPU Shepherd can see is chosen), in any layout
    index.yaml          the documents found and what happened to them
    text/<key>.md       each document's text
    skills/<slug>/      skills drafted from documents (knowledge_skills.py)

A scan finds the documents and converts each one to Markdown:

    EPUB, PDF, TXT, Markdown, HTML, DOCX    converted here (PDF with pypdf)
    AZW3, MOBI, PRC                         only counted as other copies: converting them needs Calibre

A document often comes in several formats: copies in the same folder with the same name. Only the best
copy is converted: EPUB > AZW3 > MOBI > PRC > PDF > DOCX > HTML > Markdown > TXT. One whose text matches one
already found (the same book from two bundles, under another name) is a duplicate and skipped. Ones that
can't be read (DRM, a scanned PDF with no text) are listed with the reason. When the reader gets better
(READER goes up), the ones it couldn't read are tried again at the next scan.

Older installs kept these as <data>/books, <data>/state/books.yaml, <data>/book-text and
<data>/book-skills: migrate() moves them on first use.

Pure file work: no network and no containers here. Sending documents to an app is in service.py
with connectors/openwebui.py; skills are made in knowledge_skills.py.
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
import unicodedata
import zipfile
from zlib import error as zlib_error
from html.entities import name2codepoint
from urllib.parse import unquote
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import yaml

from .config import load_yaml

READER = 2                   # goes up when reading gets better: documents it couldn't read are tried again
HOME = "knowledge"           # <data>/knowledge holds it all
# the lower, the better a copy to read a document from
RANK = {".epub": 0, ".azw3": 1, ".mobi": 2, ".prc": 3, ".pdf": 4, ".docx": 5, ".md": 6, ".markdown": 6, ".txt": 7}
CONVERTS = {".epub", ".pdf", ".docx", ".md", ".markdown", ".txt"}
TEXT = {".md", ".markdown", ".txt"}
TEXT_BOOK_BYTES = 20480      # a text file in a zip smaller than this is a readme or licence, not a book
MIN_WORDS = 200              # less text than this in an ebook: nothing to search (a scanned PDF, pictures only)
MIN_TEXT_WORDS = 20          # a note (TXT, Markdown, DOCX) can be short
# loose text files that come with ebooks but aren't documents
NOT_DOCUMENTS = re.compile(r"^(read ?me|license|licence|copying|changelog|credits|install|errata|version)\b", re.I)
SAMPLE_WORDS = 400           # words compared to find the same book under another name
UPLOAD_MB = 4096             # the biggest book the dashboard takes as an upload (streamed to disk, so only free space matters)
ZIP_MB = 65536               # the biggest zip of books it takes (unpacked into the books folder)


class BookError(Exception):
    """Why a document can't be read."""


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
    reader: int = 0              # READER when it was read

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


def _readme(path: Path) -> bool:
    """A short readme or licence beside the ebooks: not a document."""
    if path.suffix.lower() not in TEXT or not NOT_DOCUMENTS.match(path.stem):
        return False
    try:
        return path.stat().st_size < TEXT_BOOK_BYTES
    except OSError:
        return True


def find(folder: Path) -> list[tuple[str, list[str]]]:
    """The books in a folder: for each, its best copy and the other copies (paths relative to
    the folder). Copies of one book are files in the same folder with the same name (apart from
    their format and words like "retail"); the same book under other names is found by its text."""
    by_dir: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_duplicates")))
        rel = os.path.relpath(root, folder)
        for f in sorted(files):
            if Path(f).suffix.lower() in RANK and not f.startswith(".") and not _readme(Path(root) / f):
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
        raise BookError(f"{parts[-1]}: not a document (EPUB, PDF, DOCX, TXT, Markdown, AZW3, MOBI, PRC) or a zip of them")
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


def zip_has_books(path: Path) -> bool | None:
    """Whether a zip holds any books (None: it can't be opened), read from its directory only, so a
    zip of music, videos or a game isn't sent or unpacked for nothing. Short text files (a readme,
    a licence) don't count."""
    try:
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                parts = m.filename.split("/")
                if m.is_dir() or any(p.startswith(".") or p == "__MACOSX" for p in parts):
                    continue
                ext = Path(parts[-1]).suffix.lower()
                if ext in RANK and (ext not in TEXT or m.file_size >= TEXT_BOOK_BYTES):
                    return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError):
        return None
    return False


def unpack(zip_path: Path, into: Path, name: str = "") -> tuple[list[str], int]:
    """Take the books out of a zip into a folder, keeping the zip's folders. Other files are left
    out. Returns the books' paths relative to `into` and how many other files were left out.
    `name` is what to call the zip in errors (it may be saved under a temporary name)."""
    name = name or zip_path.name
    saved, other = [], 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            if sum(m.file_size for m in members) > ZIP_MB << 20:
                raise BookError(f"{name} unpacks to more than {ZIP_MB // 1024} GB")
            free = shutil.disk_usage(into if into.exists() else into.parent).free
            if sum(m.file_size for m in members) > free:
                raise BookError(f"{name}: not enough free space to unpack it")
            for m in members:
                if m.filename.startswith(("__MACOSX/", "/")):
                    continue
                try:
                    rel = safe_rel(m.filename)
                except BookError:
                    other += 1
                    continue
                ext = Path(rel).suffix.lower()
                if ext == ".zip" or (ext in TEXT and m.file_size < TEXT_BOOK_BYTES):
                    other += 1            # a zip in a zip (not opened), a readme or licence
                    continue
                dest = into / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                with z.open(m) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.replace(tmp, dest)
                saved.append(rel)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError) as err:
        raise BookError(f"{name}: can't open the zip ({err}); is it damaged, or not fully downloaded?") from err
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
        elif tag in ("td", "th"):
            self.parts.append(" ")
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


_FONT_OBFUSCATION = ("http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC")
_HTML = (".xhtml", ".html", ".htm", ".xml")
_HTML_TYPES = ("application/xhtml+xml", "text/html", "application/x-dtbook+xml")


def _local(tag) -> str:
    """An XML tag without its namespace: EPUBs in the wild use the wrong one, or none."""
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _all(root, name: str):
    return (e for e in root.iter() if _local(e.tag) == name)


def _xml(raw: bytes):
    """Parse XML that may have a BOM, HTML entities (&nbsp;) or junk around it; None if it can't be."""
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        pass
    t = raw.decode("utf-8", "replace").lstrip("﻿ \r\n\t")
    t = re.sub(r"&([a-zA-Z][a-zA-Z0-9]*);", lambda m: m.group(0) if m.group(1) in ("amp", "lt", "gt", "quot", "apos")
               else f"&#{name2codepoint[m.group(1)]};" if m.group(1) in name2codepoint else "", t)
    t = re.sub(r"&(?![a-zA-Z#][a-zA-Z0-9]*;)", "&amp;", t)
    try:
        return ElementTree.fromstring(t[t.find("<"):].encode())
    except ElementTree.ParseError:
        return None


def _decode(raw: bytes) -> str:
    """A chapter's bytes as text, in the encoding it says it is in (UTF-8 if it doesn't say)."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace")
    m = re.search(rb"""(?:encoding|charset)\s*=\s*["']?([A-Za-z0-9_\-]+)""", raw[:1024])
    try:
        return raw.decode(m.group(1).decode() if m else "utf-8")
    except (LookupError, UnicodeDecodeError):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("cp1252", "replace")


def read_epub(path: Path) -> tuple[str, str, str]:
    """An EPUB's title, author and text, its chapters in reading order. Copes with what EPUBs in the
    wild get wrong: names in the package that differ from the files (escaped, other case), a broken
    or missing package file, and chapters left out of the reading order."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as err:
        raise BookError(f"not a readable EPUB: damaged or not fully copied ({err})") from err
    with z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        lower = {n.lower(): n for n in names}

        def member(href: str) -> str | None:
            for n in (href, unquote(href)):
                n = posixpath.normpath(n).lstrip("/")
                if n in names or n.lower() in lower:
                    return n if n in names else lower[n.lower()]
            return None

        def data(name: str) -> bytes:
            try:
                return z.read(name)
            except (KeyError, zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError,
                    zlib_error) as err:
                raise _Unreadable(name) from err

        encrypted = set()
        if "META-INF/encryption.xml" in names:            # fonts may be obfuscated; text must not be encrypted
            enc = _xml(data("META-INF/encryption.xml"))
            for d in (_all(enc, "encrypteddata") if enc is not None else ()):
                method = next(_all(d, "encryptionmethod"), None)
                ref = next(_all(d, "cipherreference"), None)
                if ref is not None and (method is None or method.get("Algorithm") not in _FONT_OBFUSCATION):
                    encrypted.add(member(ref.get("URI", "")) or ref.get("URI", ""))
        opf = opf_path = None
        container = _xml(data("META-INF/container.xml")) if "META-INF/container.xml" in names else None
        root = next(_all(container, "rootfile"), None) if container is not None else None
        if root is not None and member(root.get("full-path", "")):
            opf_path = member(root.get("full-path", ""))
        else:
            opf_path = next((n for n in names if n.lower().endswith(".opf")), None)
        if opf_path:
            opf = _xml(data(opf_path))
        title = author = ""
        order: list[str] = []
        if opf is not None:
            base = posixpath.dirname(opf_path)
            items = {}
            for i in _all(opf, "item"):
                href = i.get("href", "")
                m = member(posixpath.join(base, href)) if href else None
                if m:
                    items[i.get("id")] = (m, (i.get("media-type") or "").lower())
            for r in _all(opf, "itemref"):
                if r.get("idref") in items and items[r.get("idref")][0] not in order:
                    order.append(items[r.get("idref")][0])
            if not order:                                  # no reading order: the pages in the manifest
                order = [m for m, t in items.values() if t in _HTML_TYPES or m.lower().endswith(_HTML)]
            title = next((e.text or "" for e in _all(opf, "title") if (e.text or "").strip()), "").strip()
            author = next((e.text or "" for e in _all(opf, "creator") if (e.text or "").strip()), "").strip()
        pages = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")) and not n.startswith("META-INF/")]
        if not order:                                      # no usable package file: every page, by name
            order = sorted(pages)
        if encrypted & (set(order) | set(pages)):
            raise BookError("protected by DRM")
        chapters, bad = [], 0
        for name in order:
            try:
                t = html_text(_decode(data(name)))
            except _Unreadable:
                bad += 1
                continue
            if t:
                chapters.append(t)
        text = "\n\n".join(chapters)
        if len(text.split()) < MIN_WORDS:                 # chapters outside the reading order
            rest = [n for n in sorted(pages) if n not in order]
            for name in rest:
                try:
                    t = html_text(_decode(data(name)))
                except _Unreadable:
                    bad += 1
                    continue
                if t:
                    chapters.append(t)
            text = "\n\n".join(chapters)
        if bad and not chapters:
            raise BookError("not a readable EPUB: its pages can't be unpacked (damaged, or not fully copied)")
        if "META-INF/rights.xml" in names and _garbled(text):
            raise BookError("protected by DRM")
    if _garbled(text):
        raise BookError("its text is scrambled: protected by DRM, or damaged")
    return title, author, text


class _Unreadable(Exception):
    pass


def _garbled(text: str) -> bool:
    """Encrypted or binary bytes read as text: mostly replacement and control characters."""
    sample = text[:20000]
    if len(sample) < 500:
        return False
    odd = sum(1 for c in sample if c == "�" or (unicodedata.category(c) in ("Cc", "Co", "Cn") and c not in "\n\t"))
    return odd > len(sample) * 0.05


def read_docx(path: Path) -> tuple[str, str, str]:
    """A Word document's title, author and text: its paragraphs, headings kept."""
    try:
        with zipfile.ZipFile(path) as z:
            body = _xml(z.read("word/document.xml"))
            core = _xml(z.read("docProps/core.xml")) if "docProps/core.xml" in z.namelist() else None
    except (zipfile.BadZipFile, KeyError, OSError, NotImplementedError, RuntimeError, zlib_error) as err:
        raise BookError(f"not a readable Word document ({err.__class__.__name__})") from err
    if body is None:
        raise BookError("not a readable Word document")
    out = []
    for para in _all(body, "p"):
        style = next((e.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "")
                      for e in _all(para, "pstyle")), "")
        t = "".join(e.text or "" if _local(e.tag) == "t" else "\t" if _local(e.tag) == "tab" else ""
                    for e in para.iter()).strip()
        if not t:
            continue
        level = 1 if re.match(r"(?i)title", style) else int(m.group(1)) if (m := re.match(r"(?i)heading ?(\d)", style)) else 0
        out.append("#" * min(level, 6) + " " + t if level else t)
    title = author = ""
    if core is not None:
        title = next((e.text or "" for e in _all(core, "title")), "").strip()
        author = next((e.text or "" for e in _all(core, "creator")), "").strip()
    return title, author, "\n\n".join(out)


def read_pdf(path: Path) -> tuple[str, str, str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import DependencyError, PdfReadError
    except ImportError as err:            # pragma: no cover - a dependency
        raise BookError("reading PDFs needs the pypdf package") from err
    try:
        r = PdfReader(str(path))
        if r.is_encrypted and not r.decrypt(""):
            raise BookError("protected by a password or DRM")
        try:
            meta = r.metadata or {}
        except Exception:  # noqa: BLE001 - broken metadata: still read the text
            meta = {}
        n = len(r.pages)
    except BookError:
        raise
    except DependencyError as err:            # an AES-encrypted PDF with pypdf installed without crypto
        raise BookError("its encryption needs the cryptography package (pip install 'pypdf[crypto]')") from err
    except (PdfReadError, OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        raise BookError(f"not a readable PDF ({err.__class__.__name__})") from err
    pages, bad = [], 0
    for i in range(n):                        # one broken page doesn't lose the book
        try:
            pages.append(pdf_page_text(r.pages[i].extract_text() or ""))
        except Exception:  # noqa: BLE001 - pypdf raises many kinds of errors on odd pages
            bad += 1
    if n and bad == n:
        raise BookError("not a readable PDF (no page could be read)")
    text = "\n\n".join(p for p in pages if p)
    if len(text.split()) < MIN_WORDS and n > 2:
        raise BookError("no text in it (a scanned PDF needs OCR first)")
    return _meta(meta, "/Title"), _meta(meta, "/Author"), text


def _meta(meta, key: str) -> str:
    try:
        return str(meta.get(key) or "").replace("\x00", "").strip()
    except Exception:  # noqa: BLE001
        return ""


def pdf_page_text(text: str) -> str:
    """A PDF page's text, tidied: runs of spaces collapsed, words hyphenated across a line break
    joined again ("exam-" + "ple"), blank lines squeezed. Lines and indents are kept (code needs them)."""
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = re.sub(r"(?<=\S)[ \t]+", " ", text)          # indents (code) kept
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read(path: Path) -> tuple[str, str, str]:
    """A document's title, author and text (Markdown). Raises BookError when it can't be read."""
    ext = path.suffix.lower()
    short = MIN_WORDS
    if ext == ".epub":
        title, author, text = read_epub(path)
    elif ext == ".pdf":
        title, author, text = read_pdf(path)
    elif ext == ".docx":
        title, author, text = read_docx(path)
        short = MIN_TEXT_WORDS
    elif ext in TEXT:
        text = _decode(path.read_bytes())
        title, author, short = "", "", MIN_TEXT_WORDS
    else:
        raise BookError(f"{ext[1:].upper()} needs Calibre to convert; add an EPUB or PDF copy instead")
    if len(text.split()) < short:
        raise BookError("almost no text in it" + (" (only pictures?)" if ext == ".epub" else ""))
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

def migrate(root: Path) -> list[str]:
    """Move an older install's books (<data>/books, state/books.yaml, book-text, book-skills) into
    <data>/knowledge. Renames on one disk, so even a big library moves at once. A books folder that
    can't be renamed (another disk mounted there) stays where it is and is used from there.
    Returns what was moved, for the log."""
    home, moved = root / HOME, []
    old_index, new_index = root / "state" / "books.yaml", home / "index.yaml"
    data = {}
    if old_index.exists() and not new_index.exists():
        try:
            data = load_yaml(old_index.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
    for old, new in ((root / "book-text", home / "text"), (root / "book-skills", home / "skills"),
                     (root / "books", home / "documents")):
        if not old.is_dir() or new.exists():
            continue
        if old.name == "books" and data.get("folder") not in ("", None, str(old)):
            continue                                     # another folder is used: leave this one alone
        try:
            home.mkdir(parents=True, exist_ok=True)
            os.rename(old, new)
            moved.append(f"{old} -> {new}")
            if data.get("folder") == str(old):
                data["folder"] = ""
        except OSError:
            if old.name == "books":
                data["folder"] = str(old)
    if old_index.exists() and not new_index.exists():
        home.mkdir(parents=True, exist_ok=True)
        new_index.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        old_index.unlink(missing_ok=True)
        moved.append(f"{old_index} -> {new_index}")
    elif not new_index.exists() and data.get("folder"):
        home.mkdir(parents=True, exist_ok=True)
        new_index.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return moved


def size(path: Path) -> tuple[int, int]:
    """How many files a folder holds and their bytes (0, 0 if it isn't there)."""
    n = total = 0
    for root, _dirs, names in os.walk(path):
        for f in names:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
                n += 1
            except OSError:
                pass
    return n, total


class Library:
    """The documents folder, the documents found in it and their text."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.home = self.root / HOME
        self.moved = migrate(self.root)
        self.index_file = self.home / "index.yaml"
        self.text_dir = self.home / "text"
        self.skills_dir = self.home / "skills"
        self.default_folder = self.home / "documents"
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
        return Path(data.get("folder") or self.default_folder)

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
            raise BookError(f"GPU Shepherd can't see the documents folder {folder}")
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
            if prev and prev.stamp == stamp and (self.text_path(key).exists() if prev.status in ("ready", "duplicate")
                                                 else prev.reader >= READER):
                b.title, b.author, b.status, b.reason = prev.title, prev.author, prev.status, prev.reason
                b.words, b.fingerprint, b.sent, b.reader = prev.words, prev.fingerprint, prev.sent, prev.reader
            else:
                progress(i, len(found), rel)
                b.reader, b.sent = READER, prev.sent if prev else {}
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
