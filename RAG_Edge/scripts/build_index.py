#!/usr/bin/env python3
"""
Build a persistent RAG index from a text file extracted from a PDF.

  pdftotext who_dengue_2009.pdf who_dengue_2009.txt
  python3 build_index.py docs/who_dengue_2009.txt

Pipeline: clean the text -> split into numbered sections -> chunk ->
embed with llama-server (--embedding, port 8082) -> store in SQLite.
Only the standard library and `requests` are needed.
"""
import argparse
import re
import sqlite3
import sys
import time
from array import array

import requests

EMBED_URL = "http://127.0.0.1:8082/v1/embeddings"

# Some embedding models were trained with a prefix on each side.
# The same table lives in ask_dengue.py, so both scripts agree.
PREFIXES = {
    "bge-small":      ("Represent this sentence for searching relevant passages: ", ""),
    "minilm":         ("", ""),
    "e5-small":       ("query: ", "passage: "),
    "nomic-embed":    ("search_query: ", "search_document: "),
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
}

# Embedded at build time and stored, so ask_dengue.py can check that the
# server it talks to is running the same embedding model.
PROBE = "Dengue is transmitted by Aedes mosquitoes."

RUNNING_HEADER = "Dengue: Guidelines for diagnosis, treatment, prevention and control"
SECTION_RE = re.compile(r"^(\d\.\d+(?:\.\d+)*)\s+([A-Z].{2,90})$")


# ─── 1. Cleaning: undo what the PDF layout did to the text ──────────
def clean_lines(raw):
    lines = []
    for line in raw.replace("\f", "\n").splitlines():
        s = line.strip()
        if s == RUNNING_HEADER or re.fullmatch(r"CHAPTER \d", s):
            continue                                  # page headers
        if re.fullmatch(r"[ivxlc]+|\d{1,3}", s):
            continue                                  # page number
        if re.fullmatch(r"[-–—\s]{10,}.*|.*-{10,}\s*\d+", s):
            continue                                  # table-of-contents leaders
        lines.append(s)
    return lines


def to_paragraphs(lines):
    """Join wrapped lines into paragraphs; keep headings on their own."""
    paras, buf = [], []

    def flush():
        if buf:
            paras.append(" ".join(buf))
            buf.clear()

    for s in lines:
        if not s:
            flush()
        elif SECTION_RE.match(s) or s.startswith("Chapter ") or s == "REFERENCES":
            flush()
            paras.append(s)
        elif s == "•":
            flush()
            buf.append("•")                           # bullet on its own line
        else:
            if buf and buf[-1].endswith("-") and not buf[-1].endswith(" -"):
                buf[-1] = buf[-1][:-1] + s            # re-join hyphenated words
            else:
                buf.append(s)
    flush()
    return [re.sub(r"\s+", " ", p).replace("• ", "- ") for p in paras]


# ─── 2. Structure: group paragraphs under their numbered section ────
def split_sections(paras):
    sections, current = [], {"id": "0", "title": "Front matter", "paras": []}
    chapter = ""
    for p in paras:
        if p.startswith("Chapter "):
            chapter = p
            continue
        m = SECTION_RE.match(p) or re.match(r"()(REFERENCES)$", p)
        if m:
            sections.append(current)
            current = {"id": m.group(1), "title": m.group(2).strip(),
                       "chapter": chapter, "paras": []}
        else:
            current["paras"].append(p)
    sections.append(current)
    # Reference lists match almost any query and answer none; drop them,
    # along with the front matter (acknowledgements, abbreviations).
    return [s for s in sections
            if s["paras"] and s["id"] != "0"
            and not s["title"].lower().startswith("references")]


# ─── 3. Chunking: pack paragraphs up to max_words, with overlap ─────
def chunk_section(sec, max_words, overlap_words):
    words_of = lambda t: len(t.split())
    chunks, buf = [], []
    for p in sec["paras"]:
        if buf and sum(map(words_of, buf)) + words_of(p) > max_words:
            chunks.append(" ".join(buf))
            tail = " ".join(buf).split()[-overlap_words:] if overlap_words else []
            buf = [" ".join(tail)] if tail else []
        while words_of(p) > max_words:                # one very long paragraph
            w = p.split()
            chunks.append(" ".join(buf + [" ".join(w[:max_words])]))
            buf, p = [], " ".join(w[max_words - overlap_words:])
        buf.append(p)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# ─── 4. Embedding: one HTTP call per batch of chunks ────────────────
def embed(texts, session):
    r = session.post(EMBED_URL, json={"input": texts}, timeout=600)
    r.raise_for_status()
    data = r.json()
    vecs = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    return vecs, data["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("textfile")
    ap.add_argument("--db", default="dengue.db")
    ap.add_argument("--model", default="bge-small", choices=PREFIXES)
    ap.add_argument("--max-words", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    raw = open(args.textfile, encoding="utf-8").read()
    sections = split_sections(to_paragraphs(clean_lines(raw)))
    rows = []
    for sec in sections:
        for text in chunk_section(sec, args.max_words, args.overlap):
            # Contextual header: the section title travels with the chunk,
            # so a chunk that says "these patients" still knows its topic.
            rows.append((sec["id"], sec["title"], f"{sec['id']} {sec['title']}\n{text}"))
    print(f"{len(sections)} sections -> {len(rows)} chunks "
          f"(max {args.max_words} words, overlap {args.overlap})")

    db = sqlite3.connect(args.db)
    db.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS meta;
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, section TEXT, title TEXT,
                             text TEXT, n_tokens INTEGER, emb BLOB);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks',
                                                   content_rowid='id');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)

    doc_prefix = PREFIXES[args.model][1]
    session = requests.Session()
    t0, total_tokens = time.perf_counter(), 0
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        vecs, n_tok = embed([doc_prefix + r[2] for r in batch], session)
        total_tokens += n_tok
        for (sec_id, title, text), vec in zip(batch, vecs):
            db.execute("INSERT INTO chunks (section, title, text, n_tokens, emb) "
                       "VALUES (?, ?, ?, ?, ?)",
                       (sec_id, title, text, len(text.split()),
                        array("f", vec).tobytes()))
        done = min(i + args.batch, len(rows))
        rate = done / (time.perf_counter() - t0)
        print(f"\r  embedded {done}/{len(rows)} chunks ({rate:.1f} chunks/s)",
              end="", flush=True)
    elapsed = time.perf_counter() - t0
    db.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")
    probe = embed([PROBE], session)[0][0]
    db.executemany("INSERT INTO meta VALUES (?, ?)",
                   [("model", args.model), ("dim", str(len(vecs[0]))),
                    ("max_words", str(args.max_words)),
                    ("probe", array("f", probe).tobytes().hex())])
    db.commit()
    print(f"\nIndexed {len(rows)} chunks, {total_tokens} tokens in {elapsed:.0f} s "
          f"({total_tokens / elapsed:.0f} tokens/s) -> {args.db}")


if __name__ == "__main__":
    sys.exit(main())
