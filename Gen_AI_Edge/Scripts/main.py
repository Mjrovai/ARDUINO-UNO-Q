"""
risk-classifier: Dengue risk classification on the UNO Q.

Three interfaces over the same classify() core:
  1. Bridge RPC  — on-board MCU sketch (returns float risk code).
  2. Flask POST  — /classify  for off-board JSON clients.
  3. Flask GET   — /          live HTML dashboard (polls /status).
"""

import json
import re
import socket
import struct
import threading
import time
import requests
from flask import Flask, request, jsonify, render_template_string
from arduino.app_utils import *

# ─── Container-aware host discovery ────────────────────────────────
# Inside the App Lab container, 127.0.0.1 is the *container's* loopback,
# not the UNO Q's. The default gateway in /proc/net/route is the host
# as seen from this container, which is where llama-server listens.

def _host_gateway():
    """Return the default gateway IP (the UNO Q host from inside the container)."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fields = line.strip().split()
                if fields[1] == "00000000" and int(fields[3], 16) & 2:
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except OSError:
        pass
    return "127.0.0.1"   # fallback when running outside a container

# ─── Configuration ──────────────────────────────────────────────────
LLM_HOST       = _host_gateway()
LLM_URL        = f"http://{LLM_HOST}:8081/v1/chat/completions"
LLM_HEALTH_URL = f"http://{LLM_HOST}:8081/health"
TIMEOUT_S      = 120
FLASK_PORT     = 7000

# Qwen3.5 non-thinking mode parameters (from Unsloth docs)
LLM_PARAMS = {
    "temperature":    0.7,
    "top_p":          0.8,
    "presence_penalty": 1.5,
}

SYSTEM_PROMPT = (
    "You are an environmental risk classifier for dengue surveillance. "
    "Given temperature (Celsius), humidity (%), and whether standing water "
    "is reported, output ONLY a JSON object with two fields: "
    '"risk" (one of: low, medium, high) and '
    '"reason" (one short sentence, max 20 words). '
    "Do not include any prose outside the JSON."
)

FEW_SHOTS = [
    {"role": "user",
     "content": '{"temp_c": 18.0, "humidity_pct": 40, "standing_water": false}'},
    {"role": "assistant",
     "content": '{"risk":"low","reason":"Cool and dry conditions with no water; Aedes mosquito activity unlikely."}'},
    {"role": "user",
     "content": '{"temp_c": 29.5, "humidity_pct": 82, "standing_water": true}'},
    {"role": "assistant",
     "content": '{"risk":"high","reason":"Warm humid conditions plus standing water create ideal Aedes breeding habitat."}'},
]

# Risk code mapping — float so Bridge.call() >> float works on the MCU
RISK_CODE = {"low": 0.0, "medium": 1.0, "high": 2.0}

# ─── Shared state (updated on every classification) ─────────────────
_last_status = {
    "risk":          "unknown",
    "reason":        "No reading yet.",
    "temp_c":        None,
    "humidity_pct":  None,
    "standing_water": None,
    "latency_ms":    0,
}

# ─── Core inference helpers ─────────────────────────────────────────

def strip_think_blocks(text):
    """Remove residual <think>...</think> tags that Qwen3.5 emits even with
    --reasoning off. This is a known behaviour in current llama.cpp builds."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_messages(payload):
    user_msg = {"role": "user",
                "content": json.dumps(payload, ensure_ascii=False)}
    return [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOTS, user_msg]


def call_llm(messages, max_tokens=80):
    body = {
        "model": "qwen3.5-0.8b",
        "messages": messages,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        **LLM_PARAMS,
    }
    t0 = time.perf_counter()
    r  = requests.post(LLM_URL, json=body, timeout=TIMEOUT_S)
    r.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000
    content    = r.json()["choices"][0]["message"]["content"]
    content    = strip_think_blocks(content)
    return content, latency_ms


def parse_verdict(content):
    """Defensive parse: tolerate stray code fences if the model adds them."""
    try:
        verdict = json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip().lstrip("`").rstrip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
        verdict = json.loads(cleaned)
    if verdict.get("risk") not in {"low", "medium", "high"}:
        raise ValueError(f"unexpected risk value: {verdict!r}")
    return verdict


def classify(temp_c, humidity_pct, standing_water):
    """Core inference — called by both the Bridge wrapper and Flask."""
    payload = {
        "temp_c":        float(temp_c),
        "humidity_pct":  float(humidity_pct),
        "standing_water": bool(standing_water),
    }
    messages = build_messages(payload)
    content, latency_ms = call_llm(messages)
    try:
        verdict = parse_verdict(content)
    except (json.JSONDecodeError, ValueError):
        # one retry with stricter token budget
        content, latency_ms2 = call_llm(messages, max_tokens=60)
        verdict = parse_verdict(content)
        latency_ms += latency_ms2
    verdict["latency_ms"] = round(latency_ms, 1)
    print(f"[classify] {payload} -> {verdict}")
    return verdict


# ─── Bridge wrapper (MCU-facing) ────────────────────────────────────

def classify_risk(temp_c, humidity_pct, standing_water):
    """Called by the MCU sketch via Bridge.call("classify", ...).

    Returns a float so RpcCall.result(float) works on the MCU side:
        0.0 = low   1.0 = medium   2.0 = high   -1.0 = error
    Also updates _last_status so the web dashboard stays current.
    """
    global _last_status
    verdict = classify(float(temp_c), float(humidity_pct), bool(standing_water))
    _last_status = {
        "risk":          verdict.get("risk", "unknown"),
        "reason":        verdict.get("reason", ""),
        "temp_c":        round(float(temp_c), 1),
        "humidity_pct":  round(float(humidity_pct), 1),
        "standing_water": bool(standing_water),
        "latency_ms":    verdict.get("latency_ms", 0),
    }
    return RISK_CODE.get(verdict.get("risk", "unknown"), -1.0)


# ─── Dashboard HTML ─────────────────────────────────────────────────

DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dengue Risk Monitor · UNO Q</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
           display: flex; flex-direction: column; align-items: center;
           min-height: 100vh; padding: 2rem 1rem; }
    h1   { font-size: 1.3rem; letter-spacing: .05em; color: #94a3b8; margin-bottom: 2rem; }
    #card {
      width: 100%; max-width: 420px; border-radius: 1.5rem;
      padding: 2.5rem 2rem; text-align: center;
      transition: background .6s, box-shadow .6s;
      background: #1e293b; box-shadow: 0 0 0 0 transparent;
    }
    #card.low    { background: #14532d; box-shadow: 0 0 40px 4px #22c55e55; }
    #card.medium { background: #713f12; box-shadow: 0 0 40px 4px #eab30855; }
    #card.high   { background: #7f1d1d; box-shadow: 0 0 40px 4px #ef444455; }
    #risk-label  { font-size: 4rem; font-weight: 800; letter-spacing: .04em;
                   text-transform: uppercase; margin-bottom: .5rem; }
    #reason      { font-size: 1rem; color: #cbd5e1; margin-bottom: 2rem; min-height: 2.5em; }
    .metrics     { display: grid; grid-template-columns: 1fr 1fr 1fr;
                   gap: .75rem; margin-bottom: 1.5rem; }
    .metric      { background: #ffffff18; border-radius: .75rem; padding: .75rem .5rem; }
    .metric .val { font-size: 1.4rem; font-weight: 700; }
    .metric .lbl { font-size: .7rem; color: #94a3b8; text-transform: uppercase; }
    #meta        { font-size: .75rem; color: #64748b; }
    #dot         { display: inline-block; width: .5rem; height: .5rem;
                   border-radius: 50%; background: #64748b;
                   margin-right: .3rem; vertical-align: middle; }
    #dot.live    { background: #22c55e; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  </style>
</head>
<body>
  <h1>🦟 Dengue Risk Monitor &nbsp;·&nbsp; Arduino UNO Q</h1>
  <div id="card">
    <div id="risk-label">—</div>
    <div id="reason">Waiting for first reading…</div>
    <div class="metrics">
      <div class="metric"><div class="val" id="temp">—</div>
                          <div class="lbl">°C</div></div>
      <div class="metric"><div class="val" id="hum">—</div>
                          <div class="lbl">Humidity %</div></div>
      <div class="metric"><div class="val" id="water">—</div>
                          <div class="lbl">Water</div></div>
    </div>
    <div id="meta"><span id="dot"></span><span id="ts">connecting…</span></div>
  </div>

  <script>
    const EMOJI = { low: "🟢", medium: "🟡", high: "🔴", unknown: "⚪" };

    async function refresh() {
      try {
        const r = await fetch("/status");
        const d = await r.json();

        const card = document.getElementById("card");
        card.className = ["low","medium","high"].includes(d.risk) ? d.risk : "";

        document.getElementById("risk-label").textContent =
          (EMOJI[d.risk] || "⚪") + " " + (d.risk || "unknown").toUpperCase();
        document.getElementById("reason").textContent = d.reason || "";
        document.getElementById("temp").textContent =
          d.temp_c        !== null ? d.temp_c.toFixed(1)        : "—";
        document.getElementById("hum").textContent =
          d.humidity_pct  !== null ? d.humidity_pct.toFixed(1)  : "—";
        document.getElementById("water").textContent =
          d.standing_water === null ? "—" : d.standing_water ? "YES" : "no";

        const dot = document.getElementById("dot");
        dot.className = "live";
        setTimeout(() => { dot.className = ""; }, 800);

        document.getElementById("ts").textContent =
          "Last reading: " + new Date().toLocaleTimeString() +
          "  ·  " + (d.latency_ms / 1000).toFixed(1) + " s inference";
      } catch (e) {
        document.getElementById("ts").textContent = "⚠ fetch error — retrying";
      }
    }

    refresh();
    setInterval(refresh, 5000);   // poll every 5 s
  </script>
</body>
</html>
"""

# ─── Flask app (off-board interfaces) ───────────────────────────────

flask_app = Flask(__name__)

@flask_app.route("/healthz", methods=["GET"])
def healthz():
    try:
        r = requests.get(LLM_HEALTH_URL, timeout=5)
        return jsonify({"flask": "ok", "llm": r.json()}), 200
    except Exception as e:
        return jsonify({"flask": "ok", "llm_error": str(e)}), 503

@flask_app.route("/classify", methods=["POST"])
def classify_endpoint():
    """Off-board JSON clients: POST sensor readings, get full verdict."""
    p = request.get_json(force=True)
    required = {"temp_c", "humidity_pct", "standing_water"}
    if not required.issubset(p):
        return jsonify({"error": f"missing fields: {required - set(p)}"}), 400
    verdict = classify(p["temp_c"], p["humidity_pct"], p["standing_water"])
    # keep _last_status in sync when called from HTTP too
    global _last_status
    _last_status = {
        "risk":          verdict.get("risk", "unknown"),
        "reason":        verdict.get("reason", ""),
        "temp_c":        round(float(p["temp_c"]), 1),
        "humidity_pct":  round(float(p["humidity_pct"]), 1),
        "standing_water": bool(p["standing_water"]),
        "latency_ms":    verdict.get("latency_ms", 0),
    }
    return jsonify(verdict), 200

@flask_app.route("/status", methods=["GET"])
def status_endpoint():
    """Return the latest classification result as JSON (polled by the dashboard)."""
    return jsonify(_last_status), 200

@flask_app.route("/", methods=["GET"])
def dashboard():
    """Serve the live HTML dashboard."""
    return render_template_string(DASHBOARD_HTML)


def run_flask():
    # threaded=False — the model serves one request at a time anyway
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=False)


# ─── Main entry ─────────────────────────────────────────────────────

# Expose classify_risk() to the MCU sketch via Bridge RPC
Bridge.provide("classify", classify_risk)

# Start Flask in a background thread
threading.Thread(target=run_flask, daemon=True).start()

print(f"[init] llama-server target : {LLM_URL}")
print(f"[init] Bridge registered   : classify -> classify_risk")
print(f"[init] Flask listening     : 0.0.0.0:{FLASK_PORT}")
print(f"[init] Dashboard           : http://<UNO_Q_IP>:{FLASK_PORT}/")


def loop():
    # Main loop is idle — all work is event-driven:
    # Bridge calls arrive from the MCU, HTTP requests arrive via Flask.
    time.sleep(1)


App.run(user_loop=loop)