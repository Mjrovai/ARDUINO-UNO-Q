#!/usr/bin/env python3
"""
Ask questions about the WHO dengue guidelines, answered by a local SLM
grounded on chunks retrieved from the SQLite index built by build_index.py.

  python3 ask_dengue.py                          # interactive
  python3 ask_dengue.py "What are the warning signs of severe dengue?"
  python3 ask_dengue.py --mode hybrid --k 3 "..."
  python3 ask_dengue.py --no-rag "..."           # same question, no retrieval

Needs two llama-server processes: the SLM on port 8081 and the
embedding model on port 8082 (optionally a reranker on port 8083).
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from array import array
from operator import mul

import requests

CHAT_URL = "http://127.0.0.1:8081/v1/chat/completions"
EMBED_URL = "http://127.0.0.1:8082/v1/embeddings"
RERANK_URL = "http://127.0.0.1:8083/v1/rerank"
MODEL = "qwen3.5-0.8b"

PREFIXES = {
    "bge-small":      ("Represent this sentence for searching relevant passages: ", ""),
    "minilm":         ("", ""),
    "e5-small":       ("query: ", "passage: "),
    "nomic-embed":    ("search_query: ", "search_document: "),
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
}

PROBE = "Dengue is transmitted by Aedes mosquitoes."   # same as build_index.py

SYSTEM = (
    "You answer questions using ONLY the numbered context passages from the "
    "WHO dengue guidelines (2009). Be brief: at most four sentences. "
    "Cite the passages you used, like [1] or [2]. Only if no passage "
    "answers the question, reply with just: The guidelines do not cover this."
)

session = requests.Session()


# ─── Loading the index ──────────────────────────────────────────────
class Index:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        meta = dict(self.db.execute("SELECT key, value FROM meta"))
        self.model = meta["model"]
        self.query_prefix = PREFIXES[self.model][0]
        self.rows = []                                # (id, section, title, text, vec)
        for cid, sec, title, text, blob in self.db.execute(
                "SELECT id, section, title, text, emb FROM chunks"):
            vec = array("f")
            vec.frombytes(blob)
            self.rows.append((cid, sec, title, text, vec))
        self.by_id = {r[0]: r for r in self.rows}
        self.check_model(meta.get("probe"))

    def check_model(self, probe_hex):
        """Vectors from different models are not comparable: fail loudly."""
        if not probe_hex:
            return
        stored = array("f")
        stored.frombytes(bytes.fromhex(probe_hex))
        r = session.post(EMBED_URL, json={"input": [PROBE]}, timeout=60)
        r.raise_for_status()
        now = r.json()["data"][0]["embedding"]
        if len(now) != len(stored) or sum(map(mul, now, stored)) < 0.99:
            sys.exit(f"The embedding server on {EMBED_URL} is not running the model "
                     f"this index was built with ({self.model}).")

    def embed_query(self, question):
        r = session.post(EMBED_URL, json={"input": [self.query_prefix + question]},
                         timeout=60)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    # Vectors are unit length, so cosine similarity is a plain dot product.
    def vector_search(self, qvec, k):
        scored = [(sum(map(mul, qvec, r[4])), r[0]) for r in self.rows]
        scored.sort(reverse=True)
        return scored[:k]

    # SQLite's FTS5 ranks by BM25: exact words, no embeddings involved.
    def keyword_search(self, question, k):
        words = re.findall(r"[A-Za-z0-9]{3,}", question)
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self.db.execute(
            "SELECT rowid, bm25(chunks_fts) FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k)).fetchall()
        return [(-score, cid) for cid, score in rows]   # bm25: lower is better


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: merge ranked lists without comparing scores."""
    fused = {}
    for ranking in rankings:
        for rank, (_, cid) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(((s, cid) for cid, s in fused.items()), reverse=True)


def rerank(question, index, candidates, k):
    docs = [index.by_id[cid][3] for _, cid in candidates]
    r = session.post(RERANK_URL, json={"query": question, "documents": docs,
                                       "top_n": k}, timeout=120)
    r.raise_for_status()
    results = r.json()["results"]
    return [(res["relevance_score"], candidates[res["index"]][1]) for res in results]


# ─── Retrieval: vector, keyword, or both ────────────────────────────
def retrieve(question, index, k=3, mode="vector", use_rerank=False):
    t = {}
    t0 = time.perf_counter()
    qvec = index.embed_query(question)
    t["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pool = 4 * k if (mode == "hybrid" or use_rerank) else k
    vec_hits = index.vector_search(qvec, pool)
    best_cosine = vec_hits[0][0]
    if mode == "hybrid":
        hits = rrf([vec_hits, index.keyword_search(question, pool)])
    else:
        hits = vec_hits
    t["search"] = time.perf_counter() - t0

    if use_rerank:
        t0 = time.perf_counter()
        hits = rerank(question, index, hits[:pool], k)
        t["rerank"] = time.perf_counter() - t0
    return hits[:k], best_cosine, t


# ─── Generation: stream the answer, keep llama-server's timings ─────
def generate(messages, max_tokens=200, cache_prompt=True):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.3, "top_p": 0.8, "presence_penalty": 1.5,
               "stream": True, "cache_prompt": cache_prompt}
    t0 = time.perf_counter()
    first_token, answer, timings = None, [], {}
    with session.post(CHAT_URL, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.encoding = "utf-8"          # SSE has no charset; requests would guess Latin-1
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings", timings)
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0]["delta"].get("content") or ""
            if delta and first_token is None:
                first_token = time.perf_counter() - t0
            print(delta, end="", flush=True)
            answer.append(delta)
    print()
    return "".join(answer), first_token or 0.0, time.perf_counter() - t0, timings


def ask(question, index, k=3, mode="vector", use_rerank=False,
        threshold=0.0, no_rag=False):
    if no_rag:
        messages = [{"role": "system", "content": "Be brief: at most four sentences."},
                    {"role": "user", "content": question}]
        _, ttft, total, tm = generate(messages)
        print(f"\n[no RAG] first token {ttft:.1f} s, total {total:.1f} s, "
              f"prompt {tm.get('prompt_n', '?')} tokens")
        return

    hits, best, t = retrieve(question, index, k, mode, use_rerank)
    if best < threshold:
        print(f"The guidelines do not cover this. (best similarity {best:.2f} "
              f"< threshold {threshold:.2f}, no SLM call)")
        return

    context = "\n\n".join(f"[{i}] (section {index.by_id[cid][1]}) {index.by_id[cid][3]}"
                          for i, (_, cid) in enumerate(hits, 1))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    _, ttft, total, tm = generate(messages)

    print("\nSources:")
    for i, (score, cid) in enumerate(hits, 1):
        _, sec, title, _, _ = index.by_id[cid]
        print(f"  [{i}] {sec} {title[:60]}  (score {score:.3f})")
    extra = f", rerank {t['rerank']:.2f} s" if "rerank" in t else ""
    print(f"[{mode}, k={k}] embed {t['embed']:.2f} s, search {t['search']:.2f} s{extra}, "
          f"first token {ttft:.1f} s, total {total:.1f} s | "
          f"prompt {tm.get('prompt_n', '?')} tokens "
          f"@ {tm.get('prompt_per_second', 0):.1f} t/s, "
          f"answer {tm.get('predicted_n', '?')} tokens "
          f"@ {tm.get('predicted_per_second', 0):.1f} t/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--db", default="dengue.db")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()

    index = Index(args.db)
    print(f"Loaded {len(index.rows)} chunks (embeddings: {index.model})\n")
    opts = dict(k=args.k, mode=args.mode, use_rerank=args.rerank,
                threshold=args.threshold, no_rag=args.no_rag)
    if args.question:
        ask(" ".join(args.question), index, **opts)
        return
    try:
        while True:
            q = input("\nQuestion> ").strip()
            if q:
                ask(q, index, **opts)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    sys.exit(main())
