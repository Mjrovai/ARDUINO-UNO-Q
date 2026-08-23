"""
main.py — the agent loop. Sends the conversation + tool schemas to
llama-server, dispatches any tool_calls, feeds results back, repeats
until the model returns a plain-text answer. Exposed over HTTP so you
can reach it from a terminal or another device on the network.
"""

import json
import socket
import struct
import threading
import time

import requests
from flask import Flask, request, jsonify
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

LLM_URL = f"http://{_host_gateway()}:8081/v1"
LLM_HEALTH_URL = f"http://{_host_gateway()}:8081/health"
FLASK_PORT = 7000

client = OpenAI(base_url=LLM_URL, api_key="not-needed")
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
        "description": "List files and folders in the agent's workspace directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path inside the workspace. Default '.'."}
        }},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the agent's workspace directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path to the file inside the workspace."}
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
        "description": "Draw a pattern on the UNO Q's onboard 8x13 LED matrix. "
                       "Use 'off' to clear it.",
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

SYSTEM_PROMPT = (
    "You are a helpful assistant running locally on an Arduino UNO Q. "
    "You have tools to inspect the board's system info, read files in "
    "your workspace, do arithmetic, and control the onboard LED and LED "
    "matrix. Use a tool whenever the answer depends on information you "
    "don't already know (system state, file contents, exact arithmetic) "
    "or requires a physical action. Otherwise, just answer directly. "
    "Use as few tools as possible: most questions need zero or one call. "
    "Never call the same tool twice with the same arguments. "
    "As soon as the tool results contain enough to answer, stop calling "
    "tools and reply in one or two plain sentences."
)

MAX_TURNS = 6


def run_agent(user_message):
    """Run the agent loop. Returns {"answer": str, "trace": [...]}.

    The trace is returned as well as printed, so HTTP clients see the
    tool calls and not just the conclusion. Watching the decisions is
    most of the point of this chapter.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    trace = []
    started = time.time()

    for turn in range(MAX_TURNS):
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
            max_tokens=512,
        )
        llm_ms = int((time.time() - t0) * 1000)
        print(f"[turn {turn}] model replied in {llm_ms} ms", flush=True)
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Plain-text answer: the agent is done.
            return {"answer": msg.content, "trace": trace,
                    "total_ms": int((time.time() - started) * 1000)}

        # The model wants to call one or more tools. Note `or ""` — on a
        # pure tool-call turn `content` is None, and some OpenAI-compatible
        # servers reject a null content field when it's replayed back.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name not in DISPATCH:
                result = json.dumps({"error": f"unknown tool '{name}'"})
            else:
                try:
                    result = DISPATCH[name](**args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})

            print(f"[turn {turn}] tool call: {name}({args})", flush=True)
            print(f"[turn {turn}] result: {result}", flush=True)
            trace.append({"turn": turn, "tool": name, "args": args,
                          "result": result, "llm_ms": llm_ms})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return {"answer": "(gave up after too many tool-call turns — see MAX_TURNS)",
            "trace": trace, "total_ms": int((time.time() - started) * 1000)}

# ─── Flask app (the interface you talk to) ─────────────────────────

flask_app = Flask(__name__)


# A single page, served as a plain string — no template files, no Jinja
# variables, no build step. It POSTs to /ask and renders the trace next to
# the answer, so the tool calls are visible to whoever is watching.

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Board Agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body { font: 16px system-ui, sans-serif; max-width: 46rem; margin: 2rem auto;
        padding: 0 1rem; color: #222; }
 h1 { font-size: 1.3rem; }
 .row { display: flex; gap: .5rem; }
 input { flex: 1; padding: .6rem; font: inherit; border: 1px solid #bbb;
         border-radius: 6px; }
 button { padding: .6rem 1.1rem; font: inherit; border: 0; border-radius: 6px;
          background: #2b6; color: #fff; cursor: pointer; }
 button:disabled { background: #999; cursor: default; }
 .step { background: #f4f4f2; padding: .5rem .8rem; border-radius: 6px;
         font-family: ui-monospace, monospace; font-size: .8rem;
         margin: .4rem 0; word-break: break-word; }
 .answer { background: #eef6ff; padding: .9rem; border-radius: 6px;
           border-left: 4px solid #2b6; margin-top: .6rem; white-space: pre-wrap; }
 .meta { color: #777; font-size: .8rem; }
</style></head><body>
<h1>Board Agent</h1>
<div class="row">
  <input id="q" placeholder="Ask the board something..." autofocus>
  <button id="b">Ask</button>
</div>
<div id="out"></div>
<script>
var q = document.getElementById('q');
var b = document.getElementById('b');
var out = document.getElementById('out');

function line(cls, text) {
  var d = document.createElement('div');
  d.className = cls;
  d.textContent = text;
  out.appendChild(d);
}

function run() {
  var question = q.value.trim();
  if (!question) return;
  b.disabled = true;
  out.textContent = '';
  line('meta', 'Thinking...');
  fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question })
  })
  .then(function (r) { return r.json(); })
  .then(function (d) {
    out.textContent = '';
    (d.trace || []).forEach(function (s) {
      line('step', '[turn ' + s.turn + '] tool call: ' + s.tool +
           '(' + JSON.stringify(s.args) + ')');
      line('step', '[turn ' + s.turn + '] result: ' + s.result);
    });
    line('answer', d.answer || '(no answer)');
    if (d.total_ms) line('meta', (d.total_ms / 1000).toFixed(1) + 's');
  })
  .catch(function (e) {
    out.textContent = '';
    line('meta', 'Request failed: ' + e);
  })
  .then(function () { b.disabled = false; });
}

b.onclick = run;
q.onkeydown = function (e) { if (e.key === 'Enter') run(); };
</script></body></html>"""


@flask_app.route("/", methods=["GET"])
def index():
    return INDEX_HTML


@flask_app.route("/healthz", methods=["GET"])
def healthz():
    try:
        r = requests.get(LLM_HEALTH_URL, timeout=5)
        return jsonify({"flask": "ok", "llm": r.json()}), 200
    except Exception as e:
        return jsonify({"flask": "ok", "llm_error": str(e)}), 503


@flask_app.route("/ask", methods=["POST"])
def ask_endpoint():
    p = request.get_json(force=True)
    question = (p.get("question") or "").strip()
    if not question:
        return jsonify({"error": "missing 'question'"}), 400
    return jsonify(run_agent(question)), 200


def run_flask():
    # threaded=False because the model handles one request at a time anyway
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=False)


# ─── Main entry: start Flask, then hand the main thread to App.run ──

threading.Thread(target=run_flask, daemon=True).start()

print(f"[init] llama-server target: {LLM_URL}", flush=True)
print(f"[init] agent ready, Flask on :{FLASK_PORT}", flush=True)


def loop():
    # The main loop is idle. All work is event-driven, arriving as HTTP
    # requests that Flask handles on its own thread.
    time.sleep(1)


App.run(user_loop=loop)
