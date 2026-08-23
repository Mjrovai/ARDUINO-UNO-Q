#!/usr/bin/env python3
"""ask.py — command-line client for the Board Agent.

Standard library only, so it runs with the board's system Python. It does
not import anything from the app; it just talks to the HTTP endpoint.

    python3 ask.py "what is 234 times 17, minus 9?"
"""

import json
import sys
import urllib.request

URL = "http://localhost:7000/ask"


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: python3 ask.py "your question"')
        return 1

    req = urllib.request.Request(
        URL,
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Generous timeout: a multi-turn answer on an 0.8B model is not fast.
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.load(r)

    for step in payload.get("trace", []):
        print(f'[turn {step["turn"]}] tool call: {step["tool"]}({step["args"]})')
        print(f'[turn {step["turn"]}] result: {step["result"]}')
    print(payload.get("answer", ""))
    if "total_ms" in payload:
        print(f'\n({payload["total_ms"] / 1000:.1f}s)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
