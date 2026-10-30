#!/usr/bin/env python3
"""
Measure retrieval quality with a small hand-labeled question set.

  python3 eval_retrieval.py --db dengue.db --mode vector
  python3 eval_retrieval.py --db dengue.db --mode hybrid --rerank

For each question we know which section of the WHO guidelines holds the
answer. hit@k = fraction of questions whose section appears in the top k.
The out-of-scope questions show where a similarity threshold should go.
No SLM is involved: only the embedding server (and the reranker) must run.
"""
import argparse
import time

from ask_dengue import Index, retrieve

QUESTIONS = [
    ("How long is the extrinsic incubation period of the virus in the mosquito?", ["1.4.4"]),
    ("How long does clinically significant plasma leakage last in the critical phase?", ["2.1.2"]),
    ("Which active ingredients should mosquito repellents contain?", ["3.2.6"]),
    ("How is the Breteau index calculated?", ["5.2.2.2"]),
    ("When are ovitraps better than larval surveys?", ["5.2.2.9"]),
    ("What should be done with discarded tires?", ["3.2.1.3"]),
    ("Which patients should be referred for in-hospital management?", ["2.3.2.2"]),
    ("How should specimens for virus isolation be kept before transport?", ["4.3.1"]),
    ("How many dengue virus serotypes are there, and what genus are they?", ["1.4.1"]),
    ("Can copepods be used to control Aedes aegypti?", ["3.2.7.2"]),
    ("What is the incubation period of dengue in humans?", ["1.4.2"]),
    ("What is a lethal ovitrap?", ["3.2.8.2"]),
    ("Which painkillers should be avoided in dengue patients?", ["2.3.2.1", "2.3.3.3"]),
    ("How many DALYs were lost to dengue globally in 2001?", ["1.2"]),
    # The case definitions in 1.1.6 and Textbox 2.3.3.3 also answer this one.
    ("Does a positive tourniquet test help diagnose dengue?", ["2.1.1", "1.1.6", "2.3.3.3"]),
    # Portuguese: does an English-only embedding model still find the passage?
    ("Quanto tempo dura o extravasamento de plasma na fase crítica?", ["2.1.2"]),
    ("Quais repelentes devem ser usados contra o mosquito?", ["3.2.6"]),
]

OUT_OF_SCOPE = [
    "How do I configure Wi-Fi on the Arduino UNO Q?",
    "What is the capital of France?",
    "How is malaria treated?",
    "Give me a recipe for sourdough bread.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="dengue.db")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    index = Index(args.db)
    hits = {1: 0, 3: 0, 5: 0}
    mrr, in_scope_best, t_total = 0.0, [], 0.0
    for q, expected in QUESTIONS:
        t0 = time.perf_counter()
        results, best, _ = retrieve(q, index, k=5, mode=args.mode,
                                    use_rerank=args.rerank)
        t_total += time.perf_counter() - t0
        in_scope_best.append(best)
        sections = [index.by_id[cid][1] for _, cid in results]
        rank = next((i for i, s in enumerate(sections, 1) if s in expected), None)
        for k in hits:
            hits[k] += rank is not None and rank <= k
        mrr += 1 / rank if rank else 0
        if args.verbose:
            print(f"{'ok ' if rank == 1 else rank or '-':>3} {best:.2f}  {q[:60]}  {sections}")

    n = len(QUESTIONS)
    print(f"{index.model:15s} {args.mode:6s} rerank={args.rerank!s:5s} "
          f"hit@1 {hits[1]}/{n}  hit@3 {hits[3]}/{n}  hit@5 {hits[5]}/{n}  "
          f"MRR {mrr / n:.2f}  ({t_total / n * 1000:.0f} ms/query)")

    out_best = [retrieve(q, index, k=1)[1] for q in OUT_OF_SCOPE]
    print(f"  best cosine, in scope: min {min(in_scope_best):.2f} "
          f"median {sorted(in_scope_best)[n // 2]:.2f} | "
          f"out of scope: {', '.join(f'{b:.2f}' for b in out_best)}")


if __name__ == "__main__":
    main()
