#!/usr/bin/env python3
import re
import sys
import requests

URL = "http://127.0.0.1:8081/completion"

def strip_think_blocks(text: str) -> str:
    # Remove blocos <think>...</think> (mesmo que vazios)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def ask_qwen(prompt: str, n_predict: int = 256) -> str:
    resp = requests.post(
        URL,
        json={
            "prompt": prompt,
            "n_predict": n_predict,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # /completion retorna objeto com "content"
    raw = data.get("content", "")
    return strip_think_blocks(raw)

def main():
    if len(sys.argv) > 1:
        # modo linha de comando: qwen "pergunta aqui"
        prompt = " ".join(sys.argv[1:])
        answer = ask_qwen(prompt)
        print(answer)
    else:
        # modo REPL simples
        print("Qwen CLI (UNO-Q via llama-server). Ctrl+C para sair.\n")
        try:
            while True:
                try:
                    prompt = input("> ").strip()
                except EOFError:
                    break
                if not prompt:
                    continue
                answer = ask_qwen(prompt)
                print()
                print(answer)
                print()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
