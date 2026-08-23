"""
tools.py — the agent's toolbox. Each public function here corresponds to one
entry in the `tools=[...]` schema sent to llama-server.

Every function returns a JSON string, and every failure is returned rather
than raised. That's deliberate: a tool that raises kills the agent loop; a
tool that returns {"error": "..."} hands the model something it can read and
recover from on the next turn.
"""

import ast
import json
import operator
import os
import platform
import shutil
from pathlib import Path

# ─── get_system_info ────────────────────────────────────────────────
# No model-controlled arguments, so nothing to validate: the commands
# below are the only ones that can ever run, whatever the user types.

def get_system_info():
    """Return basic OS/CPU/RAM/disk facts about the board.

    Every size is a plain number in GB, not a formatted string. That is
    deliberate: this output is read by a model, not a person. Handed
    "1.4G free of 9.8G", a small model has to parse the string and reason
    about units before it can compare anything -- and it often gets that
    wrong. Handed disk_free_gb: 1.4, the comparison is one step.
    """
    def _pretty_name():
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return platform.platform()

    def _mem_gb():
        total = avail = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1048576
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 1048576
        except (OSError, ValueError):
            pass
        return total, avail

    du = shutil.disk_usage("/")
    mem_total, mem_avail = _mem_gb()

    return json.dumps({
        "os": _pretty_name(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_free_gb": round(mem_avail, 1) if mem_avail else None,
        "memory_total_gb": round(mem_total, 1) if mem_total else None,
        "disk_free_gb": round(du.free / 2**30, 1),
        "disk_total_gb": round(du.total / 2**30, 1),
    })


# ─── Workspace-scoped file access ───────────────────────────────────
# These take a path from the model, so they get a boundary.

# Resolve the workspace relative to this file, never as an absolute host
# path: arduino-app-cli runs the app from /app inside a container, so
# "/home/arduino/ArduinoApps/board-agent/workspace" does not exist there.
# tools.py lives at <app>/python/tools.py, so its grandparent is the app root.
APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path(
    os.environ.get("AGENT_WORKSPACE", APP_ROOT / "workspace")
).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
MAX_READ_BYTES = 4096


def _resolve_in_workspace(path):
    """Resolve `path` against WORKSPACE_ROOT and refuse anything that escapes it."""
    candidate = (WORKSPACE_ROOT / path).resolve()
    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise ValueError(f"path '{path}' is outside the workspace")
    return candidate


def list_files(path="."):
    """List files and folders inside the agent's workspace directory."""
    try:
        target = _resolve_in_workspace(path)
        if not target.exists():
            return json.dumps({"error": f"'{path}' does not exist in the workspace"})
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        if not entries:
            # Say so explicitly. A bare [] gives the model nothing to act on,
            # and it will often just call the tool again -- see Section 10.
            return json.dumps({"path": path, "entries": [],
                               "note": "this directory is empty"})
        return json.dumps({"path": path, "entries": entries})
    except (ValueError, OSError) as e:
        return json.dumps({"error": str(e)})


def read_file(path):
    """Read a text file from the agent's workspace directory (capped size)."""
    try:
        target = _resolve_in_workspace(path)
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file in the workspace"})
        with target.open("r", errors="replace") as fh:
            content = fh.read(MAX_READ_BYTES)
            truncated = fh.read(1) != ""
        return json.dumps({"path": path, "content": content, "truncated": truncated})
    except (ValueError, OSError) as e:
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

MAX_EXPONENT = 64


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent {right} is above the limit of {MAX_EXPONENT}")
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def calculate(expression):
    """Safely evaluate an arithmetic expression (+ - * / ** and parentheses only)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})
