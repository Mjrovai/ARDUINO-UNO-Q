#!/usr/bin/env python3
"""
End-to-end latency of RAG on the UNO Q for different top-k values.

  python3 rag_bench.py --db dengue.db --k 1 2 3 5

For each k, asks the same three questions and reports the prompt size
(all tokens, and the part not served from llama-server's prompt cache),
the time to the first answer token, and the total time.
"""
import argparse
import contextlib
import io
import statistics

from ask_dengue import Index, SYSTEM, generate, retrieve

QUESTIONS = [
    "Which painkillers should be avoided in dengue patients?",
    "How is the Breteau index calculated?",
    "How long does clinically significant plasma leakage last?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="dengue.db")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--mode", default="vector")
    ap.add_argument("--cold", action="store_true",
                    help="disable llama-server's prompt cache (every token is processed)")
    args = ap.parse_args()

    index = Index(args.db)
    print(f"{args.db}: {len(index.rows)} chunks")
    print(f"{'k':>3} {'prompt tok':>10} {'processed':>9} {'1st token s':>11} {'total s':>8}")
    # Warm-up: the first request also caches the system prompt.
    with contextlib.redirect_stdout(io.StringIO()):
        generate([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": "Hello"}], max_tokens=1)
    for k in args.k:
        rows = []
        for q in QUESTIONS:
            hits, _, _ = retrieve(q, index, k=k, mode=args.mode)
            context = "\n\n".join(f"[{i}] (section {index.by_id[cid][1]}) {index.by_id[cid][3]}"
                                  for i, (_, cid) in enumerate(hits, 1))
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}]
            with contextlib.redirect_stdout(io.StringIO()):
                _, ttft, total, tm = generate(messages, max_tokens=120,
                                              cache_prompt=not args.cold)
            rows.append((tm["prompt_n"] + tm.get("cache_n", 0), tm["prompt_n"], ttft, total))
        mean = lambda i: statistics.mean(r[i] for r in rows)
        print(f"{k:>3} {mean(0):>10.0f} {mean(1):>9.0f} {mean(2):>11.1f} {mean(3):>8.1f}")


if __name__ == "__main__":
    main()
