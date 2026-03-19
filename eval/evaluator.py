"""
Evaluator — scores each agent run on multiple criteria.
Produces a structured report used in the Streamlit UI.
"""

import re


class Evaluator:
    def evaluate(
        self,
        query: str,
        code: str,
        success: bool,
        attempts: int,
        fig=None,
    ) -> dict:
        scores = {}

        # 1. Execution success (0 or 4 points)
        scores["execution"] = 4 if success else 0

        # 2. Chart produced (0 or 2 points)
        scores["chart_produced"] = 2 if fig is not None else 0

        # 3. Self-healing bonus: succeeded after retries (0–2 points)
        if success and attempts == 1:
            scores["self_healing"] = 2  # perfect first try
        elif success and attempts > 1:
            scores["self_healing"] = 1  # needed retries but recovered
        else:
            scores["self_healing"] = 0

        # 4. Code quality heuristics (0–2 points)
        cq = 0
        if "fig.update_layout" in code or "title=" in code:
            cq += 1  # has chart title / layout customisation
        if "dropna" in code or "fillna" in code:
            cq += 0  # defensive coding (no extra points, just nice to see)
        if len(code.strip().splitlines()) >= 3:
            cq += 1  # non-trivial code
        scores["code_quality"] = min(cq, 2)

        total = sum(scores.values())

        # Feedback message
        if total >= 9:
            feedback = "Excellent — chart rendered perfectly on first attempt."
        elif total >= 7:
            feedback = "Good — agent produced a valid chart, minor issues."
        elif total >= 5:
            feedback = "Partial — chart rendered after self-healing retries."
        else:
            feedback = "Failed — agent could not produce a valid chart."

        return {
            "score": total,
            "max_score": 10,
            "breakdown": scores,
            "success": success,
            "attempts": attempts,
            "feedback": feedback,
            "chart_type_detected": self._detect_chart_type(code),
        }

    @staticmethod
    def _detect_chart_type(code: str) -> str:
        patterns = {
            "bar": r"px\.bar|go\.Bar",
            "scatter": r"px\.scatter|go\.Scatter",
            "histogram": r"px\.histogram|go\.Histogram",
            "pie": r"px\.pie|go\.Pie",
            "box": r"px\.box|go\.Box",
            "heatmap": r"px\.imshow|go\.Heatmap",
            "line": r"px\.line|go\.Line",
            "scatter_matrix": r"px\.scatter_matrix",
            "violin": r"px\.violin|go\.Violin",
            "bubble": r"px\.scatter.*size=|go\.Scatter.*marker.*size",
        }
        for chart, pattern in patterns.items():
            if re.search(pattern, code):
                return chart
        return "unknown"
