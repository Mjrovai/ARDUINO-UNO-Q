#!/usr/bin/env python3
"""
What an embedding model does: turn sentences into vectors whose
dot product measures how close their meanings are.
Needs llama-server with an embedding model on port 8082.
"""
import requests

EMBED_URL = "http://127.0.0.1:8082/v1/embeddings"

sentences = [
    "Aedes aegypti mosquitoes lay their eggs in standing water.",
    "Mosquito larvae develop in water-filled containers such as old tyres.",
    "Dengue fever causes high fever and severe joint pain.",
    "The Arduino UNO Q has a quad-core Cortex-A53 processor.",
]

r = requests.post(EMBED_URL, json={"input": sentences}, timeout=60)
r.raise_for_status()
vectors = [d["embedding"] for d in r.json()["data"]]
print(f"{len(vectors)} vectors of {len(vectors[0])} numbers each")
print("first 5 numbers of sentence 1:", [round(x, 3) for x in vectors[0][:5]])

# llama-server returns unit-length vectors, so the dot product
# is the cosine similarity: 1 = same direction, 0 = unrelated.
def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

print("\nCosine similarity:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"  {cosine(vectors[i], vectors[j]):.2f}  [{i + 1}] vs [{j + 1}]")
