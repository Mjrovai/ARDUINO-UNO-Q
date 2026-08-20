# Agentic AI at the Edge:

Giving a Small Model Tools

---

## 1. Introduction

### What This Tutorial Covers

Every previous chapter gave the SLM a fixed job: classify these three numbers, describe this image. The model reasoned, but it never *decided what to do*. This chapter changes that. You'll give Qwen3.5 a small set of **tools** — functions it can choose to call — and let it decide, on its own, which ones to use and in what order to satisfy a plain-language request.

The tools are deliberately ordinary: ask the board what OS it's running, do some arithmetic, list a few files, blink the built-in LED, draw a pattern on the onboard 8×13 LED matrix. None of it requires a single wire. That's on purpose — this chapter should run in any room, on any UNO Q, with nothing attached, which makes it the one to reach for in a workshop where you can't guarantee everyone has a breadboard.

The mechanism is llama-server's native, OpenAI-compatible **tool-calling API** (`tools=[...]` in the request, `tool_calls` in the response) — the same interface real cloud LLM APIs use, not a custom prompt-and-parse scheme. If you've read the "Building Agents with SLMs" chapter of the companion *Edge ML Made Easy* book, that example routed a query through hand-written JSON classification, then called the numbers found in it — a technique the book itself flags as fragile (models struggle to reliably format the classification JSON, especially at 1B and below). The native tools API doesn't eliminate that fragility at small model sizes, but it puts the parsing burden on llama.cpp's grammar-constrained decoding instead of on hopeful prompt engineering, and it's the same shape of code you'd write against a hosted model.

By the end of this chapter you'll have a small, hardware-free agent you can talk to from a terminal, watch it decide which tool to call and why, and see it act — on the board's own files, its own vitals, and its own LEDs.

### Prerequisites

This tutorial assumes you have completed:

- [Setup](../1-Setup/README.md) — headless SSH access to the board.
- [Generative AI at the Edge](../2-Gen_AI/README.md) — `llama.cpp` built from source, a Qwen3.5-0.8B GGUF downloaded, and comfort running `llama-server` and calling it from Python with the `openai` client library.

Deliberately **not** required: [Multimodal AI at the Edge](../3-Multimodal_AI_Edge/README.md) or [GenAI Meets the Real World](../4-Gen_AI_Edge/README.md). Nothing here depends on vision or on external sensors — that's the point. Section 11 shows how those chapters' tools plug into this same agent loop later, once you want them.

> If chapter 2 is unfamiliar, work through it first — this chapter reuses its `llama-server` setup and its `openai`-client patterns without repeating them.

## 2. What Makes This "Agentic"?

The word gets used loosely. Pinned down, an agent here means a model that does three things a plain chat model doesn't:

1. **Decides whether to act at all.** Given "What's the capital of Brazil?", the right move is to just answer — no tool needed. Given "How much free disk space is there?", the model can't know that; it has to call a tool and read the result.
2. **Chooses which tool, with what arguments**, from a list it's given — not a list you hardcoded a branch for.
3. **Observes the result and decides what happens next** — call another tool, or produce a final answer. That decision can chain: a request like *"check if there's more than 1 GB free, then show a check or an X on the matrix"* requires the model to call `get_system_info`, read the number out of the result itself, and only then decide which matrix pattern to draw.

Chapter 4's `classify()` does none of this. It always reads three sensor values, always calls the model exactly once, always with the same prompt shape. That's a **pipeline** — deterministic, fast, and appropriate for its job. What follows is a **loop**:

```
 ┌──────────────────────────────────────────────────────────────┐
 │  1. Send: system prompt + tool schemas + conversation so far │
 │  2. Model replies with EITHER:                               │
 │       a) a tool_calls list  → run 3, then loop back to 1     │
 │       b) plain text         → done, show it to the user      │
 │  3. Run the requested tool(s) in Python, append the result(s)│
 │     to the conversation as role="tool" messages              │
 └──────────────────────────────────────────────────────────────┘
```

This is the same loop under the hood as any "agent framework" you may have heard of (LangChain agents, OpenAI's Assistants API, etc.) — those add retries, memory, and orchestration around it, but the core cycle above is the whole mechanism. Building it by hand, in maybe 40 lines of Python, is the point of this chapter: there's no magic to demystify once you've written the loop yourself.

### The Native Tools API vs. Hand-Rolled Classification

llama-server exposes an OpenAI-compatible `tools` parameter on `/v1/chat/completions`. You describe each tool as a JSON Schema (name, description, parameter types); the server constrains decoding so the model's function-call arguments come back as valid, parseable JSON rather than prose the model might format inconsistently. Compared to the classify-then-route pattern from the Raspberry Pi book chapter — where the model had to freehand a JSON object like `{"type": "multiplication", "numbers": [7, 8]}` and the code hoped it got the shape right — the native API moves that structural guarantee into the inference engine itself. It doesn't fix everything (Section 9 has the honest version of what still goes wrong at 0.8B), but it removes one whole category of failure.

Qwen3.5 supports this function-calling format natively (chapter 4's Going Further section flagged this same capability for a different use). It needs the chat template active — the `--jinja` flag you've used since chapter 3 — since tool-call formatting is part of the model's chat template, not a separate code path.

## 3. First Contact: Agentic Mode in the Built-in WebUI

Before writing a line of Python, it's worth seeing an agent loop run with zero code — recent llama.cpp builds ship one built into the server's own WebUI (the same one you opened in chapters 2 and 3 for chat). This is the fastest way for a room full of students to *see* tool-calling happen, watch each decision, and approve or reject it in real time, before anyone opens an editor.

> In chapters 2 and 3's WebUI bonus sections, this book told you to leave the built-in agent tools switched off ("enabling filesystem or shell tools on a server bound to `0.0.0.0` is a security footgun. Keep them off in this setup"). This section is where that changes — deliberately, briefly, and with the reasoning made explicit, not just "trust me." Read the safety note below before you flip anything on.

### Step 1 — Open the WebUI and Find the Tools Settings

With `llama-server` running (`--jinja` on, as always), open the WebUI and click the gear icon (**Settings**), then **Tools** in the left-hand list. You should see something like this:

![](./images/tools-settings.png)

A **Server** group lists tools bundled with llama-server itself — typically **Read file**, **Search files**, **Search in files**, **Run command**, **Write file**, **Edit file**, and **Runtime info** — each with its own **Enabled** and **Always allow** checkboxes. A **Browser** group adds a couple of browser-side tools. Exact names and counts vary by llama.cpp build; the shape is what matters.

### Step 2 — Understand What "Always Allow" Means Before Touching It

Every tool call the model wants to make shows up as a confirmation prompt in the chat — **unless** "Always allow" is checked for that tool, in which case it runs immediately with no human in the loop. For this first demo:

- **Leave every "Always allow" box unchecked.** You want to see and approve each call, not have the model quietly reading and writing files while you're mid-sentence explaining something else.
- Enable just **Read file**, **Search files**, and **Runtime info** for a first pass — the three read-only tools. Leave **Run command**, **Write file**, and **Edit file** off until Step 4.

### Step 3 — Set the Agentic Limits

Click **Agentic** in the same Settings list:

![](./images/agentic-settings.png)

- **Agentic turns** — the same idea as the `MAX_TURNS` cap you'll build by hand in Section 7: a hard stop on how many tool-call cycles run before the server gives up. The default (often 10) is fine to start.
- **MCP request timeout** — how long a single tool call is allowed to take before it's abandoned.
- **Mention search depth** — how many directory levels the file-search tools will descend; irrelevant for this first exercise.

### Step 4 — Watch It Work

With only the read-only tools enabled, try a prompt that can't be answered from the model's own knowledge:

> *"What files are in my home directory, and how much free disk space is there?"*

Watch the chat: a confirmation card appears asking to run a specific tool with specific arguments *before* it executes. Approve it, watch the result stream back into the conversation, and see whether the model needs a second tool call (it likely will — one for the file listing, one for disk space) before it gives you a final answer. This is Section 2's loop diagram, happening in front of you, one approved step at a time.

Once that feels familiar, go back to Settings → Tools and enable **Write file** (still with "Always allow" off), then try:

> *"Create a file called notes.txt in my home directory with a two-line note about edge AI."*

Approve the write, then confirm it from a terminal: `cat ~/notes.txt`.

### A Safety Note — Read This Before Your Next Session

The built-in tools are **not sandboxed** the way the tools you build later in this chapter will be. "Read file," "Write file," "Edit file," and especially "Run command" operate with the same OS permissions as the `arduino` user running `llama-server` — which, on this board, is close to full access. There is no path restriction, no allow-list, nothing stopping "Run command" from being asked to do something destructive if a request (or a confused model) steers it there.

Two rules for using this safely in a classroom or workshop:

1. **Keep "Always allow" off** for any tool that can write or execute, so a human approves every action before it runs. Turning it on is asking for unattended file writes and shell commands.
2. **`--host 0.0.0.0` is fine on a network you control — the board is headless, so there's no other way to reach the WebUI from a laptop's browser (same reasoning as every WebUI section in this book so far).** The thing to actually watch is *who else is on that network*. On your own classroom or workshop Wi-Fi, that's the same trust level chapters 2–4 already assume. On an untrusted network (open conference Wi-Fi, a shared public space), it stops being safe: anyone who can reach the `/v1/chat/completions` endpoint can drive the same tools through the raw API, and the confirmation prompt only exists in the WebUI — a script hitting the API directly gets no such prompt. In that situation, bind to `127.0.0.1` instead and reach the WebUI over an SSH tunnel (`ssh -L 8081:127.0.0.1:8081 arduino@<UNO_Q_IP>`, then open `http://localhost:8081/` on your laptop).

That's the trade this section is making on purpose: broad, generic, unsandboxed tools, running with your full user permissions, in exchange for zero setup. The rest of this chapter builds the opposite — five narrow, purpose-built tools, each scoped to exactly what it needs and nothing more, so every permission the agent has is one you wrote and can account for. Seeing both ends of that spectrum, in one session, is the point.

## 4. Hardware and Software Requirements

### Hardware

- Arduino UNO Q (2 GB or 4 GB — this chapter's models are small enough that either works; see [chapter 2](../2-Gen_AI/README.md#candidate-models) for the size/quality tradeoffs).
- USB-C data cable.
- Host computer with SSH.

**Nothing else.** No breadboard, no sensors, no external LEDs — every tool in this chapter uses either the Linux filesystem/CLI or the UNO Q's own onboard LED and LED matrix.

### Software (already on the UNO Q from earlier chapters)

| Tool | Purpose |
|---|---|
| `llama.cpp` built from source, Qwen3.5-0.8B GGUF | From [chapter 2](../2-Gen_AI/README.md) |
| `llama-server` reachable at `localhost:8081` | Foreground (chapter 2) or systemd service (chapter 4) — this chapter doesn't care which |
| `openai` Python client | From chapter 2 |
| `arduino-app-cli` | Build/run dual-brain apps |

### Software (installed in this chapter)

Nothing new on the Python side — `openai` from chapter 2 is all this needs. The one addition is on the MCU side: the `Arduino_LED_Matrix` library, which is bundled with the UNO Q's Zephyr core and needs no separate install.

## 5. Designing the Tool Set

Five tools, each small enough to reason about in one read:

| Tool | Description | Runs on | Notes |
|---|---|---|---|
| `get_system_info()` | OS, CPU, RAM, disk | MPU (Python, `subprocess`) | Read-only |
| `list_files(path)` | List files in a directory | MPU (Python) | Sandboxed — see below |
| `read_file(path)` | Read a text file's contents | MPU (Python) | Sandboxed, size-capped |
| `calculate(expression)` | Evaluate an arithmetic expression | MPU (Python, `ast`) | No `eval()` |
| `set_builtin_led(state)` | Turn the single onboard LED on/off | MCU (Bridge RPC) | |
| `set_led_matrix(pattern)` | Draw a named pattern on the 8×13 matrix | MCU (Bridge RPC) | `check`, `x`, `smiley`, `off`, or a digit `0`–`9` |

Two design choices worth calling out before the code:

- **The file tools are sandboxed to one directory**, not the whole filesystem. The model picks the path argument; you don't want a prompt like "read /etc/shadow" to be a live possibility just because the model got asked to. `list_files`/`read_file` resolve every path against a fixed root and reject anything that escapes it.
- **`calculate` parses an AST, it doesn't `eval()`.** A tempting shortcut is `eval(expression)` — don't. The expression comes from model output, which is influenced by whatever the user typed. Walking a restricted AST (numbers, `+ - * / **`, parentheses, unary minus — nothing else) gets you a real calculator with no code-execution surface.

This is the JSON Schema the model actually sees for two of the tools (the full list is in Section 7):

```json
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Evaluate a basic arithmetic expression and return the numeric result.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "An arithmetic expression, e.g. '234 * 17 - 9' or '(3 + 4) / 2'."
        }
      },
      "required": ["expression"]
    }
  }
}
```

```json
{
  "type": "function",
  "function": {
    "name": "set_led_matrix",
    "description": "Draw a pattern on the UNO Q's onboard 8x13 LED matrix.",
    "parameters": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "enum": ["check", "x", "smiley", "off", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
          "description": "Which pattern to display."
        }
      },
      "required": ["pattern"]
    }
  }
}
```

## 6. Implementing the Tools

### Step 1 — Project Skeleton

```bash
cd ~/ArduinoApps
arduino-app-cli app new "board-agent"
cd board-agent
```

```
board-agent/
├── app.yaml
├── python/
│   ├── main.py
│   ├── tools.py
│   └── requirements.txt
└── sketch/
    ├── sketch.ino
    └── sketch.yaml
```

`app.yaml`:

```yaml
name: Board Agent
description: "A tool-calling agent for the UNO Q — no external hardware required"
icon: 🤖
version: "1.0.0"
ports: []
bricks: []
```

`python/requirements.txt`:

```
openai==1.54.0
```

### Step 2 — The Sandboxed File and System Tools

Create `python/tools.py`. This holds every tool the Linux side can run on its own, with no Bridge call involved.

```python
"""
tools.py — the agent's toolbox. Each public function here corresponds
to one entry in the `tools=[...]` schema sent to llama-server.
"""

import ast
import json
import operator
import os
import platform
import subprocess
from pathlib import Path

# ─── get_system_info ────────────────────────────────────────────────

def get_system_info():
    """Return basic OS/CPU/RAM/disk facts about the board."""
    def run(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=5).strip()
        except Exception as e:
            return f"(unavailable: {e})"

    return json.dumps({
        "os": run("cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2"),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory": run("free -h | awk '/Mem:/ {print $3\" used / \"$2\" total\"}'"),
        "disk_root": run("df -h / | awk 'NR==2 {print $4\" free of \"$2}'"),
        "disk_home": run("df -h /home/arduino | awk 'NR==2 {print $4\" free of \"$2}'"),
    })


# ─── Sandboxed file access ──────────────────────────────────────────

SANDBOX_ROOT = Path("/home/arduino/ArduinoApps/board-agent/sandbox").resolve()
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
MAX_READ_BYTES = 4096


def _resolve_in_sandbox(path):
    """Resolve `path` relative to SANDBOX_ROOT and refuse anything that escapes it."""
    candidate = (SANDBOX_ROOT / path).resolve()
    if SANDBOX_ROOT not in candidate.parents and candidate != SANDBOX_ROOT:
        raise ValueError(f"path '{path}' is outside the sandbox")
    return candidate


def list_files(path="."):
    """List files and folders inside the agent's sandbox directory."""
    try:
        target = _resolve_in_sandbox(path)
        if not target.exists():
            return json.dumps({"error": f"'{path}' does not exist in the sandbox"})
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return json.dumps({"path": path, "entries": entries})
    except ValueError as e:
        return json.dumps({"error": str(e)})


def read_file(path):
    """Read a text file from inside the agent's sandbox directory (capped size)."""
    try:
        target = _resolve_in_sandbox(path)
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file in the sandbox"})
        content = target.read_text(errors="replace")[:MAX_READ_BYTES]
        return json.dumps({"path": path, "content": content})
    except ValueError as e:
        return json.dumps({"error": str(e)})


# ─── calculate: AST-based, no eval() ────────────────────────────────

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression element: {ast.dump(node)}")


def calculate(expression):
    """Safely evaluate an arithmetic expression (+ - * / ** and parentheses only)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})
```

> **Why `_resolve_in_sandbox` checks `.resolve()` before comparing:** a naive string check like `path.startswith(SANDBOX_ROOT)` is defeated by `../../etc/passwd`-style traversal before the path is normalized. Resolving first, then checking ancestry, closes that off.

### Step 3 — The MCU Sketch: Built-In LED and LED Matrix

The UNO Q's Zephyr core bundles `Arduino_LED_Matrix` — no separate library install needed.

`sketch/sketch.ino`:

```cpp
#include "Arduino_RouterBridge.h"
#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

// 8 rows x 13 columns, 0 = off, 1 = on
uint8_t frame[8][13];

void clearFrame() {
  memset(frame, 0, sizeof(frame));
}

void showOff()   { clearFrame(); matrix.renderBitmap(frame, 8, 13); }

void showCheck() {
  clearFrame();
  int pts[][2] = {{5,2},{6,3},{7,4},{6,5},{5,6},{4,7},{3,8},{2,9},{1,10}};
  for (auto &p : pts) frame[p[0]][p[1]] = 1;
  matrix.renderBitmap(frame, 8, 13);
}

void showX() {
  clearFrame();
  for (int i = 0; i < 8; i++) {
    int col1 = 2 + i, col2 = 10 - i;
    if (col1 < 13) frame[i][col1] = 1;
    if (col2 >= 0 && col2 < 13) frame[i][col2] = 1;
  }
  matrix.renderBitmap(frame, 8, 13);
}

void showSmiley() {
  clearFrame();
  int pts[][2] = {
    {1,3},{1,4},{1,8},{1,9},              // eyes
    {5,2},{6,3},{6,4},{6,5},{6,6},{6,7},{6,8},{6,9},{5,10}  // smile
  };
  for (auto &p : pts) frame[p[0]][p[1]] = 1;
  matrix.renderBitmap(frame, 8, 13);
}

// A minimal 3x5 font for digits 0-9, positioned starting at column 4
const uint8_t DIGIT_FONT[10][5] = {
  {0b111,0b101,0b101,0b101,0b111}, // 0
  {0b010,0b110,0b010,0b010,0b111}, // 1
  {0b111,0b001,0b111,0b100,0b111}, // 2
  {0b111,0b001,0b111,0b001,0b111}, // 3
  {0b101,0b101,0b111,0b001,0b001}, // 4
  {0b111,0b100,0b111,0b001,0b111}, // 5
  {0b111,0b100,0b111,0b101,0b111}, // 6
  {0b111,0b001,0b010,0b010,0b010}, // 7
  {0b111,0b101,0b111,0b101,0b111}, // 8
  {0b111,0b101,0b111,0b001,0b111}, // 9
};

void showDigit(int d) {
  clearFrame();
  if (d < 0 || d > 9) return;
  for (int row = 0; row < 5; row++) {
    for (int col = 0; col < 3; col++) {
      if ((DIGIT_FONT[d][row] >> (2 - col)) & 1) frame[row + 1][col + 5] = 1;
    }
  }
  matrix.renderBitmap(frame, 8, 13);
}

// ─── Bridge-exposed functions ───────────────────────────────────────

void set_builtin_led(bool state) {
  digitalWrite(LED_BUILTIN, state ? HIGH : LOW);
}

// pattern: "check" | "x" | "smiley" | "off" | "0".."9"
void set_led_matrix(String pattern) {
  if (pattern == "check")       showCheck();
  else if (pattern == "x")      showX();
  else if (pattern == "smiley") showSmiley();
  else if (pattern == "off")    showOff();
  else if (pattern.length() == 1 && isDigit(pattern[0])) showDigit(pattern.toInt());
  else                          showX();  // unrecognized pattern -> visible error state
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  matrix.begin();
  showOff();

  Bridge.begin();
  Bridge.provide("set_builtin_led", set_builtin_led);
  Bridge.provide("set_led_matrix", set_led_matrix);
}

void loop() {
  // All work is event-driven via Bridge calls from Python.
}
```

`sketch/sketch.yaml`:

```yaml
profiles:
  default:
    fqbn: arduino:zephyr:uno_q
    platforms:
      - platform: arduino:zephyr
    libraries:
      - dependency: Arduino_RPClite (0.2.1)
default_profile: default
```

> `Arduino_LED_Matrix` doesn't need a `libraries:` entry — it's bundled with the `arduino:zephyr` platform core itself, not fetched from the Library Manager. If your build complains it's missing, update the core (`arduino-app-cli system update`) rather than trying to install the library separately.

> **Test the matrix patterns before wiring them into the agent.** Start the app once with a hardcoded `set_led_matrix("smiley")` call in `setup()` and confirm each pattern actually looks right on the physical 8×13 grid — the coordinate lists above are a starting point, not guaranteed pixel-perfect on first try. Nudge the `pts[][2]` coordinates until they look right on your board before moving on.

## 7. Building the Agent Loop

### Step 1 — Tool Schemas and the Dispatch Table

`python/main.py`, part 1 — wire the Python-side tools from `tools.py` to their JSON schemas, and the two Bridge-side tools to `Bridge.call()`:

```python
"""
main.py — the agent loop. Sends the conversation + tool schemas to
llama-server, dispatches any tool_calls, feeds results back, repeats
until the model returns a plain-text answer.
"""

import json
import socket
import struct
from openai import OpenAI
from arduino.app_utils import *
import tools

# ─── Container-aware host discovery (same pattern as chapter 4) ────

def _host_gateway():
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fields = line.strip().split()
                if fields[1] == "00000000" and int(fields[3], 16) & 2:
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except OSError:
        pass
    return "127.0.0.1"

client = OpenAI(base_url=f"http://{_host_gateway()}:8081/v1", api_key="not-needed")
MODEL = "qwen3.5-0.8b"

# ─── Tool schemas sent to the model ─────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "get_system_info",
        "description": "Get the board's OS, CPU, RAM, and disk usage.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files and folders in the agent's sandbox directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path inside the sandbox. Default '.'."}
        }},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the agent's sandbox directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path to the file inside the sandbox."}
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression and return the numeric result.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. '234 * 17 - 9'"}
        }, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "set_builtin_led",
        "description": "Turn the UNO Q's single onboard LED on or off.",
        "parameters": {"type": "object", "properties": {
            "state": {"type": "boolean", "description": "true = on, false = off"}
        }, "required": ["state"]},
    }},
    {"type": "function", "function": {
        "name": "set_led_matrix",
        "description": "Draw a pattern on the UNO Q's onboard 8x13 LED matrix.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string",
                        "enum": ["check", "x", "smiley", "off",
                                 "0","1","2","3","4","5","6","7","8","9"]}
        }, "required": ["pattern"]},
    }},
]

# ─── Dispatch table: tool name -> Python callable ───────────────────

def _set_builtin_led(state):
    Bridge.call("set_builtin_led", bool(state))
    return json.dumps({"ok": True, "led": "on" if state else "off"})


def _set_led_matrix(pattern):
    Bridge.call("set_led_matrix", str(pattern))
    return json.dumps({"ok": True, "pattern": pattern})


DISPATCH = {
    "get_system_info": lambda **kw: tools.get_system_info(),
    "list_files": lambda path=".": tools.list_files(path),
    "read_file": lambda path: tools.read_file(path),
    "calculate": lambda expression: tools.calculate(expression),
    "set_builtin_led": lambda state: _set_builtin_led(state),
    "set_led_matrix": lambda pattern: _set_led_matrix(pattern),
}
```

### Step 2 — The Loop Itself

```python
SYSTEM_PROMPT = (
    "You are a helpful assistant running locally on an Arduino UNO Q. "
    "You have tools to inspect the board's system info, read files in "
    "your sandbox, do arithmetic, and control the onboard LED and LED "
    "matrix. Use a tool whenever the answer depends on information you "
    "don't already know (system state, file contents, exact arithmetic) "
    "or requires a physical action. Otherwise, just answer directly. "
    "After you have enough information, give a short final answer in plain text."
)

MAX_TURNS = 6


def run_agent(user_message, verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
            max_tokens=300,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Plain-text answer: the agent is done.
            return msg.content

        # The model wants to call one or more tools.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"[turn {turn}] tool call: {name}({args})")

            if name not in DISPATCH:
                result = json.dumps({"error": f"unknown tool '{name}'"})
            else:
                try:
                    result = DISPATCH[name](**args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})

            if verbose:
                print(f"[turn {turn}] result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "(gave up after too many tool-call turns — see MAX_TURNS)"


def loop():
    # Interactive: the App Lab log tab doubles as the agent's REPL for this chapter.
    pass


if __name__ == "__main__":
    print("Board Agent ready. Ctrl+C to exit.\n")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            print(run_agent(q))
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")

App.run(user_loop=loop)
```

Design choices worth flagging:

- **`MAX_TURNS` is a hard stop, not a suggestion.** A model that keeps calling tools without ever producing a final answer is a real failure mode at small sizes (Section 10). Capping the loop turns an infinite hang into a bounded, debuggable failure.
- **The dispatch table separates "what the model can call" from "how it's implemented."** Adding a tool later (Section 11) means adding one schema entry and one `DISPATCH` line — the loop itself never changes.
- **Tool results go back as `role="tool"` messages, not appended to the user's turn.** This is what the OpenAI-compatible format expects, and it's what lets the model tell the difference between "the user said this" and "a tool returned this."
- **This version runs as an interactive REPL**, not a Bridge-exposed function like chapter 4's `classify_risk`. There's no MCU-triggered event driving it — you're talking to it directly. Section 11 discusses wiring it to sensor events instead.

## 8. Running It: Example Interactions

Start the app and use the REPL from the log stream (`arduino-app-cli app logs . --follow` in a second terminal shows the `print()` output; the `input()` prompt itself needs an interactive terminal, so run `python3 python/main.py` directly over SSH for this chapter rather than through `arduino-app-cli app start` — Bridge still needs to be initialized for the LED tools to work, so keep the app framework import but drive it from a plain terminal session while you're developing).

**System info:**

```
> What's the free disk space on this board?
[turn 0] tool call: get_system_info({})
[turn 0] result: {"os": "Debian GNU/Linux 12 (bookworm)", ..., "disk_home": "14G free of 18G"}
There's about 14 GB free on /home out of an 18 GB partition.
```

**Calculator:**

```
> What is 234 times 17, minus 9?
[turn 0] tool call: calculate({'expression': '234 * 17 - 9'})
[turn 0] result: {"expression": "234 * 17 - 9", "result": 3969}
234 x 17 - 9 = 3969.
```

**Files:**

```
> What files are in my sandbox?
[turn 0] tool call: list_files({'path': '.'})
[turn 0] result: {"path": ".", "entries": ["notes.txt"]}
There's one file: notes.txt.
```

**Onboard LED:**

```
> Turn on the built-in LED
[turn 0] tool call: set_builtin_led({'state': True})
[turn 0] result: {"ok": true, "led": "on"}
Done — the onboard LED is on.
```

**A chained example — the interesting one:**

```
> Check the free space on /home. If there's more than 1GB free, show a
  checkmark on the matrix, otherwise show an X.
[turn 0] tool call: get_system_info({})
[turn 0] result: {..., "disk_home": "14G free of 18G"}
[turn 1] tool call: set_led_matrix({'pattern': 'check'})
[turn 1] result: {"ok": true, "pattern": "check"}
There's 14 GB free, well over 1 GB, so I've shown a checkmark on the matrix.
```

That last one is the whole chapter in miniature: no code branch decided which pattern to show. The model read a number out of a tool result, compared it against a threshold *you gave it in English, not in code*, and picked the matrix pattern itself.

## 9. Performance and Reliability

Everything measured in earlier chapters was a single inference call. An agent turn is at minimum two (the tool-call decision, then the follow-up call that reads the result), often three or four for a chained request — so expect wall-clock latency to scale roughly with the number of turns in the trace, not with a single generation's token count.

Honest expectations, not yet a full benchmark suite the way chapters 2–4 have:

- **Tool selection at 0.8B is not perfectly reliable.** This mirrors what the companion book's Raspberry Pi chapter found with the older classify-then-route approach at small model sizes: a 1B-class model sometimes picks a plausible-sounding wrong tool, calls a tool it doesn't need, or answers directly when it should have checked first. The native tools API constrains the *shape* of a tool call once the model decides to make one; it does not guarantee the model decides correctly. If tool selection feels unreliable on your board, try the 2B model using the swap pattern from [chapter 3](../3-Multimodal_AI_Edge/README.md#8-swapping-models-08b-for-speed-2b-for-depth) — larger models are consistently better at this specific skill, more than most other tasks in this book.
- **`temperature=0.3`** (lower than the 0.7 used for free-form chat in earlier chapters) is deliberate here — tool selection benefits from a more deterministic sampling distribution than storytelling does.
- **Latency compounds with turns.** A two-turn chained request on the 0.8B model should feel similar to two back-to-back chapter-2-style queries, not one — because that's what it is under the hood.

Treat this section as a starting point for your own measurements on your board, not a settled result the way chapter 4's latency table is.

## 10. Tips, Tricks, and Troubleshooting

### The Model Answers From "Knowledge" Instead of Calling a Tool

If you ask "what's the free disk space?" and get a made-up number instead of a tool call, the system prompt isn't landing. Strengthen it: add an explicit line like *"You do not know the current system state. You must call get_system_info before answering any question about it."* Small models respond better to blunt, repeated instructions than to subtle framing.

### The Model Calls a Tool That Doesn't Exist, or With the Wrong Arguments

The `DISPATCH` table's `except Exception` catch in Section 7 turns this into a tool result the model can see and recover from ("unknown tool" or a Python `TypeError` message), rather than crashing the loop. Check the `[turn N] tool call:` log line — if the tool name is subtly wrong (plural, different casing), the model is guessing at a name instead of reading the schema; tightening the tool `description` fields usually helps more than tightening the system prompt.

### Infinite Tool-Call Loops

If the model keeps calling the same tool with the same arguments and never produces a final answer, you've hit `MAX_TURNS`. Before raising the cap, check whether the tool result is actually useful to the model — a result that's too terse (or buried in a huge JSON blob) can leave the model unable to tell it already has what it needs.

### `tool_calls` Is Always Empty, Even When It Shouldn't Be

Confirm `--jinja` is on the `llama-server` command line (chapter 2's default, but easy to drop when copying a command). Without it, the chat template that formats tool-call output isn't active, and the model falls back to plain text — sometimes describing the tool call in prose instead of actually making one.

### The Matrix Shows the Wrong Shape

The coordinate lists in Section 6's sketch are a first draft, not a calibrated font. Load the sketch with one hardcoded pattern at a time and adjust the `pts[][2]` arrays against the physical grid — row 0 is the top row, column 0 is the left column, and `frame[row][col] = 1` lights that single LED.

### Sandbox Path Errors

If `list_files`/`read_file` reject a path you expected to work, remember they resolve *relative to* `SANDBOX_ROOT`, not the filesystem root — `path="notes.txt"` looks for `SANDBOX_ROOT/notes.txt`, not `/notes.txt`. That's the safety boundary working as intended, not a bug.

## 11. Going Further

### Extending to Real Sensors and Actuators

Nothing about the agent loop in Section 7 is specific to system info, files, or LEDs — it's generic over anything expressed as a `{name, description, parameters}` schema plus a Python callable. [GenAI Meets the Real World](../4-Gen_AI_Edge/README.md) already built exactly that shape for real hardware: `read_temperature()`, `read_humidity()`, and `set_led(color)` reading DHT22/button state and driving RGB LEDs over the same kind of Bridge call used here. Add three more entries to `TOOLS`, three more lines to `DISPATCH`, and the same loop that decides "check disk space, then show a checkmark" can decide "check the temperature, then turn on the red LED if it's hot" — except now *the model* is making that call, not a hardcoded `if risk_code == 2`. The mechanism doesn't care whether a tool touches a filesystem or a physical sensor; that's the whole appeal of the pattern.

### Longer Tool Chains and Memory

The `messages` list in `run_agent()` is rebuilt fresh on every call — there's no memory between separate REPL turns. A natural extension is to keep the conversation history across calls (like the multi-turn `openai` example in chapter 2), so a follow-up question like "what about /var/log instead?" resolves without repeating the full request.

### Bigger Models, More Reliable Agents

Section 9 already flagged this: if 0.8B's tool selection feels shaky for your use case, the 2B swap from chapter 3 is the first thing to try before reaching for prompt-engineering workarounds. Parameter count consistently helps agentic tasks more than most other tasks in this book.

### Where This Scales To

[QClaw](https://github.com/laurenvil/Uno-QClaw) is the far end of this spectrum on the same board: an agent with tools that write files, compile and flash Arduino sketches, and shell out — not just read state and flip an LED. That's a meaningfully larger trust boundary than anything in this chapter (a tool that can write arbitrary files or run arbitrary shell commands is a different safety conversation than `get_system_info`), and it's worth reading QClaw's tool implementations with that in mind before adopting the pattern: every tool you add is something the model — not you — decides when to invoke.

## 12. Conclusion

### What We Covered

This chapter built a tool-calling agent from scratch: five small tools (system info, sandboxed files, a safe calculator, the onboard LED, the LED matrix), a JSON Schema description of each, and a ~40-line loop that lets Qwen3.5 decide which to call, read the results, and chain multiple calls toward a final answer — using llama-server's native OpenAI-compatible tools API rather than hand-rolled JSON classification. No external hardware was required anywhere in the chapter.

### Advantages of This Approach

- **Workshop-ready.** Every reader with a bare UNO Q can follow along — no shopping list, no wiring diagrams to get wrong.
- **The mechanism generalizes.** The same loop that runs a calculator also runs chapter 4's sensor tools, unchanged, once you add the schema entries (Section 11).
- **Standard, transferable API.** The `tools`/`tool_calls` shape is the same one used by hosted LLM APIs — code written against `llama-server` here ports directly to a cloud model later.

### Limitations and Considerations

- **Small-model tool selection is genuinely unreliable sometimes.** This isn't a bug to fix so much as a property of running an agent on an 0.8B model — see Section 9 and the 2B escalation path.
- **The sandbox and the AST-based calculator are safety boundaries that matter more as the tool set grows.** Adding a tool that writes files or runs shell commands (as QClaw does) is a different risk profile than anything in this chapter — think about what a *wrong* tool call costs before adding one.
- **No memory across REPL sessions**, by design, to keep the loop itself simple — see Going Further for the extension.

### What's Next

- **Real sensors and actuators** — plug chapter 4's tools into this same loop (Section 11).
- **Longer-horizon agents** — persistent memory, more tools, more turns.
- **Multimodal tools** — a tool that calls chapter 3's vision pathway ("describe what the camera sees") is a natural sixth entry in `TOOLS`.

## 13. Resources

### Useful Resources

| Resource | URL |
|---|---|
| Generative AI at the Edge (prerequisite chapter) | [2-Gen_AI/README.md](../2-Gen_AI/README.md) |
| GenAI Meets the Real World (real sensors/actuators) | [4-Gen_AI_Edge/README.md](../4-Gen_AI_Edge/README.md) |
| Multimodal AI at the Edge (vision tools) | [3-Multimodal_AI_Edge/README.md](../3-Multimodal_AI_Edge/README.md) |
| llama.cpp tool-calling / function-calling docs | <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md> |
| Qwen3.5 function-calling guide (Unsloth) | <https://unsloth.ai/docs/models/qwen3.5> |
| *Edge ML Made Easy* — Building Agents with SLMs (companion book, earlier approach) | <https://mjrovai.github.io/EdgeML_Made_Ease_ebook/raspi/advancing_adgeai/adv_edgeai.html#building-agents-with-slms> |
| Arduino_LED_Matrix library (bundled with UNO Q Zephyr core) | <https://github.com/arduino-libraries/Arduino_LED_Matrix> |
| QClaw — agentic AI assistant on the UNO Q | <https://github.com/laurenvil/Uno-QClaw> |
| Arduino UNO Q Documentation | <https://docs.arduino.cc/hardware/uno-q> |

### References

1. Qwen Team, "Qwen3.5 Small Model Series," Alibaba Cloud, March 2026.
2. Gerganov, G., "llama.cpp: Inference of Meta's LLaMA model (and others) in pure C/C++," <https://github.com/ggml-org/llama.cpp>
3. Rovai, M., "Building Agents with SLMs," *Edge ML Made Easy*, <https://mjrovai.github.io/EdgeML_Made_Ease_ebook/>
4. Arduino, "Arduino UNO Q Product Page," <https://www.arduino.cc/product-uno-q/>
5. Laurenvil, D., "QClaw," <https://github.com/laurenvil/Uno-QClaw>

---

*Tutorial created for IESTI05 — Edge AI Machine Learning System Engineering, UNIFEI. Licensed under GNU General Public License 3.0.*
