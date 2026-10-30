#!/usr/bin/env python3
"""
A minimal RAG: a list of facts, an embedding model, a dot product,
and a small language model. No database, no framework.

  python3 rag_mini.py "When do Aedes aegypti mosquitoes bite?"
  python3 rag_mini.py --no-rag "When do Aedes aegypti mosquitoes bite?"

Needs llama-server twice: the SLM on port 8081 and an embedding
model on port 8082.
"""
import sys
import time

import requests

CHAT_URL = "http://127.0.0.1:8081/v1/chat/completions"
EMBED_URL = "http://127.0.0.1:8082/v1/embeddings"
MODEL = "qwen3.5-0.8b"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-small

# ─── The knowledge base: one fact per entry, so each entry is a chunk ──
FACTS = [
    "Dengue is a viral infection transmitted to humans by the bite of infected "
    "female Aedes mosquitoes, mainly Aedes aegypti and, to a lesser extent, Aedes albopictus.",
    "There are four dengue virus serotypes, DEN-1 to DEN-4. Recovery from one gives "
    "lifelong immunity against that serotype only.",
    "A second infection with a different serotype increases the risk of severe dengue.",
    "After an infected bite, symptoms usually appear after an incubation period of 4 to 10 days.",
    "Aedes aegypti bites mainly during the day, with peaks in the early morning and "
    "in the late afternoon before dusk.",
    "Aedes aegypti lays its eggs on the damp inner walls of containers just above the "
    "water line. The eggs can survive dry for months and hatch when flooded.",
    "Common breeding sites are water-storage tanks, flower-pot saucers, discarded tyres, "
    "bottles, and any container that collects rainwater.",
    "The most effective prevention is removing standing water: cover, empty, or clean "
    "water containers every week.",
    "Inside the mosquito, the virus needs 8 to 12 days before it can be transmitted "
    "(the extrinsic incubation period); warmer temperatures shorten it.",
    "Typical dengue symptoms are high fever, severe headache, pain behind the eyes, "
    "muscle and joint pain, nausea, vomiting, and rash.",
    "Warning signs of severe dengue include abdominal pain, persistent vomiting, "
    "bleeding gums, lethargy or restlessness, and a rapid drop in platelet count.",
    "There is no specific antiviral treatment for dengue. Paracetamol is used for fever; "
    "aspirin and ibuprofen should be avoided because they can aggravate bleeding.",
    "Repellents containing DEET, IR3535, or icaridin protect exposed skin from mosquito bites.",
    "Small fish and predatory copepods that eat mosquito larvae have been used for "
    "biological control in water-storage containers.",
    "Ovitraps are small containers where mosquitoes lay eggs; counting the eggs is a "
    "cheap way to monitor Aedes populations.",
]

session = requests.Session()


def embed(texts):
    r = session.post(EMBED_URL, json={"input": texts}, timeout=120)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def chat(messages):
    r = session.post(CHAT_URL, json={
        "model": MODEL, "messages": messages, "max_tokens": 150,
        "temperature": 0.3, "top_p": 0.8, "presence_penalty": 1.5}, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip(), data["timings"]


# ─── Step 1: index the facts once (embed every chunk) ───────────────
t0 = time.perf_counter()
FACT_VECTORS = embed(FACTS)
print(f"Indexed {len(FACTS)} facts in {time.perf_counter() - t0:.1f} s\n")


def rag(question, k=3):
    # Step 2: embed the question with the same model
    t0 = time.perf_counter()
    q = embed([QUERY_PREFIX + question])[0]
    # Step 3: retrieve the k most similar facts
    ranked = sorted(zip(FACTS, FACT_VECTORS), key=lambda fv: dot(q, fv[1]), reverse=True)
    context = [f for f, _ in ranked[:k]]
    t_retrieve = time.perf_counter() - t0
    for f, v in ranked[:k]:
        print(f"  {dot(q, v):.2f}  {f[:70]}...")

    # Step 4: augment the prompt with the retrieved facts
    messages = [
        {"role": "system", "content": "Answer in at most three sentences, using only "
                                      "the facts provided. If they do not contain the "
                                      "answer, say you don't know."},
        {"role": "user", "content": "Facts:\n- " + "\n- ".join(context)
                                    + f"\n\nQuestion: {question}"},
    ]
    # Step 5: generate
    answer, tm = chat(messages)
    print(f"\n{answer}\n")
    print(f"retrieval {t_retrieve:.2f} s | prompt {tm['prompt_n']} tokens in "
          f"{tm['prompt_ms'] / 1000:.1f} s | answer {tm['predicted_n']} tokens in "
          f"{tm['predicted_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--no-rag":
        answer, tm = chat([{"role": "user", "content": " ".join(args[1:])}])
        print(f"{answer}\n\nprompt {tm['prompt_n']} tokens, answer {tm['predicted_n']} "
              f"tokens in {(tm['prompt_ms'] + tm['predicted_ms']) / 1000:.1f} s")
    else:
        rag(" ".join(args) or "When do Aedes aegypti mosquitoes bite?")
