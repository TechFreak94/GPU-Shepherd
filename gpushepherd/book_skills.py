"""Agent skills made from books: the book's methods, steps and checklists, in the open Agent
Skills format (a folder with SKILL.md, https://agentskills.io) that Hermes Agent reads.

A knowledge base lets a chat model look facts up in a book; a skill tells an agent what to *do*
and when. Making one reads the whole book with the user's own model server, a part at a time
(each part must fit the chat model's context): first notes on what each part teaches you to do,
then one SKILL.md from all the notes. The chapters go beside it in references/, so the agent can
read the details when it uses the skill.

    <data>/book-skills/<name>/SKILL.md          the draft, which the user can edit
    <data>/book-skills/<name>/references/*.md   the book's chapters
Nothing reaches an agent until the user installs the draft: then the folder is copied to the
agent's skills folder (Hermes Agent: <its data folder>/skills/books/<name>).

`make` takes the chat call as a function, so this module has no network code of its own.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable

MAX_PART_WORDS = 12000
MIN_PART_WORDS = 1500
NOTES_WORDS = 350            # asked-for length of each part's notes
SKILLS_FOLDER = "skills/books"   # inside Hermes Agent's data folder

NOTES_PROMPT = """You are turning a book into a skill for an AI agent: instructions it can follow.
Below is part {n} of {total} of the book "{name}".

Write notes, at most {words} words, on what this part teaches the reader to DO:
- methods, procedures and step-by-step techniques (keep the order of steps)
- rules of thumb, checklists, formulas and decision criteria
- common mistakes and how to avoid them
Skip stories, history, praise and anything that is not actionable. Write terse bullet points.
If this part has nothing actionable, answer only: NOTHING.

--- PART {n} ---
{text}"""

SKILL_PROMPT = """You are writing a SKILL.md file: instructions an AI agent loads when a task needs
what the book "{name}" teaches. Below are notes on every part of the book.

Write the file in Markdown, at most about 1200 words, exactly in this shape:

---
name: {slug}
description: <one sentence, under 300 characters, starting "Use when" and naming the tasks and
  situations this book's methods apply to>
---

# <a short title>

## When to use
<bullets: the tasks and questions this skill is for>

## How to do it
<the book's main methods as numbered steps, grouped under ### headings>

## Checklists and rules of thumb
<bullets>

## Mistakes to avoid
<bullets>

## More detail
The book's chapters are in references/ beside this file; read the one that fits when a step needs detail.

Use only what the notes say; don't invent advice.

--- NOTES ---
{notes}"""


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:60].strip("-") or "book"


def part_words(ctx: int) -> int:
    """How many words of the book fit in one request to a model with `ctx` tokens of context,
    leaving room for the prompt and the answer (about 1.4 tokens a word)."""
    budget = max(ctx - 3000, 0) / 1.4 if ctx else MAX_PART_WORDS
    return int(min(max(budget, MIN_PART_WORDS), MAX_PART_WORDS))


def parts(text: str, words: int) -> list[str]:
    """The book in pieces of at most `words` words, cut between paragraphs where possible."""
    out, cur, n = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        w = len(para.split())
        if not w:
            continue
        if n and n + w > words:
            out.append("\n\n".join(cur))
            cur, n = [], 0
        while w > words:                                  # one paragraph longer than a part
            ws = para.split()
            out.append(" ".join(ws[:words]))
            para, w = " ".join(ws[words:]), w - words
        cur.append(para)
        n += w
    if cur:
        out.append("\n\n".join(cur))
    return out


def chapters(text: str) -> list[tuple[str, str]]:
    """The book cut at its top headings: (title, text). A book without headings is one chapter."""
    heads = list(re.finditer(r"^(#{1,2}) (.+)$", text, re.M))
    top = min((len(h.group(1)) for h in heads[1:]), default=1)
    cuts = [h for h in heads[1:] if len(h.group(1)) == top]      # the first heading is the book's title
    if not cuts:
        return [("Book", text)]
    out, start, title = [], 0, "Front matter"
    for h in cuts:
        if text[start:h.start()].strip():
            out.append((title, text[start:h.start()]))
        start, title = h.start(), h.group(2).strip()
    out.append((title, text[start:]))
    return out


def clean(answer: str) -> str:
    """A model's answer without its thinking and code fences."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    answer = answer.split("</think>")[-1].strip()
    m = re.match(r"^```[a-z]*\n(.*)\n```$", answer, re.S)
    return (m.group(1) if m else answer).strip()


def check(skill_md: str, name: str) -> str:
    """The SKILL.md with the frontmatter an agent needs, whatever the model wrote."""
    body = skill_md.strip()
    fm = re.match(r"^---\n(.*?)\n---\n?", body, re.S)
    desc = ""
    if fm:
        d = re.search(r"^description:\s*(.+(?:\n  .+)*)", fm.group(1), re.M)
        desc = re.sub(r"\s+", " ", d.group(1)).strip().strip('"\'') if d else ""
        body = body[fm.end():].lstrip()
    desc = desc or f"Use when a task needs the methods from the book {name}."
    desc = desc.replace('"', "'")[:500]
    return f'---\nname: {slug(name)}\ndescription: "{desc}"\n---\n\n{body}\n'


def make(name: str, text: str, chat: Callable[[str], str], ctx: int,
         progress: Callable[[str], None] = lambda s: None, stop: Callable[[], bool] = lambda: False) -> str:
    """The SKILL.md for a book, written by the model behind `chat` (prompt -> answer)."""
    pieces = parts(text, part_words(ctx))
    notes = []
    for i, p in enumerate(pieces, 1):
        if stop():
            raise InterruptedError("stopped")
        progress(f"Reading part {i} of {len(pieces)}")
        got = clean(chat(NOTES_PROMPT.format(n=i, total=len(pieces), name=name, words=NOTES_WORDS, text=p)))
        if got and got.strip().upper() != "NOTHING":
            notes.append(f"## Part {i}\n{got}")
    if not notes:
        raise ValueError("the model found nothing in the book to turn into steps")
    joined = "\n\n".join(notes)
    limit = part_words(ctx)
    while len(joined.split()) > limit:                  # too many notes for one request: shorten them
        if stop():
            raise InterruptedError("stopped")
        progress("Shortening the notes")
        groups = parts(joined, limit)
        if len(groups) == 1:
            joined = " ".join(joined.split()[:limit])
            break
        joined = "\n\n".join(clean(chat(NOTES_PROMPT.format(n=i, total=len(groups), name=name,
                                                           words=NOTES_WORDS, text=g)))
                             for i, g in enumerate(groups, 1))
    progress("Writing the skill")
    return check(clean(chat(SKILL_PROMPT.format(name=name, slug=slug(name), notes=joined))), name)


def write(folder: Path, skill_md: str, text: str) -> None:
    """The skill folder: SKILL.md and the book's chapters in references/."""
    refs = folder / "references"
    if refs.exists():
        shutil.rmtree(refs)
    refs.mkdir(parents=True)
    (folder / "SKILL.md").write_text(skill_md, encoding="utf-8")
    for i, (title, body) in enumerate(chapters(text), 1):
        (refs / f"{i:02d}-{slug(title)[:40]}.md").write_text(body.strip() + "\n", encoding="utf-8")


def install(draft: Path, skills_root: Path) -> Path:
    """Copy a draft into an agent's skills folder, replacing an older copy of it."""
    dest = skills_root / SKILLS_FOLDER / draft.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(draft, dest)
    return dest
