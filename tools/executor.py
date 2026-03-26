"""
Self-healing code executor.
Runs generated Python code in a sandboxed local namespace.
On failure, captures the traceback and triggers the fix loop (max 3 attempts).

Features:
- Thread-based execution with 30-second timeout
- Error telemetry: failed executions are appended to errors.jsonl in the project root
"""

import os
import json
import traceback
import threading
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


MAX_ATTEMPTS = 3

# Path to the error telemetry log (project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERRORS_LOG = os.path.join(_PROJECT_ROOT, "errors.jsonl")


def _append_error_log(entry: dict):
    """Append a single error record to errors.jsonl."""
    try:
        with open(ERRORS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # telemetry must never crash the app


class SelfHealingExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def execute(self, code: str, log_callback=None, meta: dict | None = None) -> dict:
        """
        Execute code in a safe namespace with df pre-loaded.

        Args:
            code:         Python code string to execute.
            log_callback: Optional callable(type_, msg) for live log updates.
            meta:         Optional dict with {"dataset": ..., "query": ...}
                          used for error telemetry.

        Returns:
            {"fig": ..., "error": ..., "success": bool, "namespace": ...}
        """
        namespace = {
            "df": self.df.copy(),
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
        }

        exec_exception: list = []  # mutable container for thread result

        def _run():
            try:
                exec(code, namespace)  # noqa: S102
            except Exception:
                exec_exception.append(traceback.format_exc())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=30)

        # ── Timeout check ──────────────────────────────────────────────────────
        if thread.is_alive():
            msg = "Code execution timed out after 30 seconds."
            if log_callback:
                log_callback("error", f"⏱️ {msg}")
            _append_error_log({
                "timestamp": datetime.now().isoformat(),
                "dataset":   (meta or {}).get("dataset", "unknown"),
                "query":     (meta or {}).get("query", "unknown"),
                "error_message": msg,
                "code_snippet":  code[:400],
            })
            return {"fig": None, "error": msg, "success": False, "namespace": namespace}

        # ── Execution error ────────────────────────────────────────────────────
        if exec_exception:
            error_tb = exec_exception[0]
            if log_callback:
                log_callback("error", f"💥 Execution error: {error_tb.splitlines()[-1]}")
            _append_error_log({
                "timestamp":     datetime.now().isoformat(),
                "dataset":       (meta or {}).get("dataset", "unknown"),
                "query":         (meta or {}).get("query", "unknown"),
                "error_message": error_tb.splitlines()[-1],
                "code_snippet":  code[:400],
            })
            return {"fig": None, "error": error_tb, "success": False, "namespace": namespace}

        # ── Missing fig variable ───────────────────────────────────────────────
        fig = namespace.get("fig")
        if fig is None:
            msg = "Code ran but did not produce a `fig` variable."
            if log_callback:
                log_callback("error", f"⚠️ {msg}")
            _append_error_log({
                "timestamp":     datetime.now().isoformat(),
                "dataset":       (meta or {}).get("dataset", "unknown"),
                "query":         (meta or {}).get("query", "unknown"),
                "error_message": msg,
                "code_snippet":  code[:400],
            })
            return {"fig": None, "error": msg, "success": False, "namespace": namespace}

        return {"fig": fig, "error": None, "success": True, "namespace": namespace}
