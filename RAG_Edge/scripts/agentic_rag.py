#!/usr/bin/env python3
"""
Agentic RAG: the SLM decides whether to search the WHO guidelines,
use a calculator, both, or neither, through llama-server's native
tool-calling API (start llama-server with --jinja).

  python3 agentic_rag.py "A survey found 37 positive containers in 250 houses. What is the Breteau index?"
  python3 agentic_rag.py --force-tool "..."     # first reply must be a tool call
"""
import ast
import json
import operator
import sys
import time

import requests

from ask_dengue import Index, retrieve

CHAT_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "qwen3.5-0.8b"
MAX_TURNS = 4

index = Index("dengue.db")

TOOLS = [
    {"type": "function", "function": {
        "name": "search_guidelines",
        "description": "Search the WHO dengue guidelines (diagnosis, treatment, "
                       "mosquito control, surveillance indices). Returns relevant passages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '37 * 100 / 250'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}},
            "required": ["expression"]}}},
]

SYSTEM = ("You are a dengue assistant. Use search_guidelines for anything about "
          "dengue or mosquitoes, and calculate for any arithmetic. Answer general "
          "questions directly. Keep answers short.")


# ─── The tools ──────────────────────────────────────────────────────
def search_guidelines(query):
    hits, _, _ = retrieve(query, index, k=2, mode="hybrid")
    return "\n\n".join(f"(section {index.by_id[cid][1]}) {index.by_id[cid][3]}"
                       for _, cid in hits)


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expression):
    """Arithmetic only: walk the syntax tree instead of calling eval()."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    try:
        return str(round(ev(ast.parse(expression, mode="eval").body), 4))
    except Exception as e:
        return f"error: {e}"


DISPATCH = {"search_guidelines": search_guidelines, "calculate": calculate}


# ─── The agent loop ─────────────────────────────────────────────────
def run(question, force_tool=False):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    t_start = time.perf_counter()
    for turn in range(MAX_TURNS):
        # "required" makes the first reply a tool call; later turns choose freely.
        choice = "required" if force_tool and turn == 0 else "auto"
        r = requests.post(CHAT_URL, json={
            "model": MODEL, "messages": messages, "tools": TOOLS,
            "tool_choice": choice,
            "max_tokens": 200, "temperature": 0.3, "top_p": 0.8,
            "presence_penalty": 1.5}, timeout=900)
        r.raise_for_status()
        data = r.json()
        msg, tm = data["choices"][0]["message"], data["timings"]
        print(f"  [turn {turn}] prompt {tm['prompt_n']} new tokens, "
              f"{tm['predicted_n']} generated, {(tm['prompt_ms'] + tm['predicted_ms']) / 1000:.1f} s")
        calls = msg.get("tool_calls") or []
        if not calls:
            print(f"\n{msg.get('content', '').strip()}")
            break
        messages.append(msg)
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            result = DISPATCH[name](**args) if name in DISPATCH else "unknown tool"
            print(f"  [turn {turn}] tool call: {name}({args}) -> {result[:80]!r}")
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result})
    print(f"\nTotal: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    force = bool(args) and args[0] == "--force-tool"
    question = " ".join(args[1:] if force else args)
    if question:
        run(question, force_tool=force)
    else:
        try:
            while True:
                q = input("\n> ").strip()
                if q:
                    run(q, force_tool=force)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
