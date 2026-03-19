"""
Self-healing code executor.
Runs generated Python code in a sandboxed local namespace.
On failure, captures the traceback and triggers the fix loop (max 3 attempts).
"""

import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


MAX_ATTEMPTS = 3


class SelfHealingExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def execute(self, code: str, log_callback=None) -> dict:
        """
        Execute code in a safe namespace with df pre-loaded.
        Returns: {"fig": ..., "error": ..., "success": bool, "namespace": ...}
        """
        namespace = {
            "df": self.df.copy(),
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
        }

        try:
            exec(code, namespace)  # noqa: S102
        except Exception:
            error_tb = traceback.format_exc()
            if log_callback:
                log_callback("error", f"💥 Execution error: {error_tb.splitlines()[-1]}")
            return {"fig": None, "error": error_tb, "success": False, "namespace": namespace}

        fig = namespace.get("fig")
        if fig is None:
            msg = "Code ran but did not produce a `fig` variable."
            if log_callback:
                log_callback("error", f"⚠️ {msg}")
            return {"fig": None, "error": msg, "success": False, "namespace": namespace}

        return {"fig": fig, "error": None, "success": True, "namespace": namespace}
