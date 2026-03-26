"""
Code generator tool — Azure OpenAI backed.
Handles: code generation (with conversation memory), self-healing fixes,
         AI insight summaries, query suggestions, and NL→SQL.

Features:
- Optional stream_callback for live token streaming
- Optional cost_tracker for recording token usage
"""

import json
from typing import Callable, Optional

from openai import AzureOpenAI

SYSTEM_PROMPT = """You are an expert Python data analyst. Your job is to write clean,
executable Python code using Pandas and Plotly Express (or Plotly Graph Objects)
to answer the user's data question.

Rules:
- The dataframe is already loaded as `df` — do NOT load or import any data files.
- Import only: pandas as pd, plotly.express as px, plotly.graph_objects as go, numpy as np (if needed).
- Your code MUST assign the final Plotly figure to a variable named exactly `fig`.
- Do NOT call fig.show() — the app will render it.
- Use `df.copy()` if you modify the dataframe.
- Prefer Plotly Express for simple charts; use Graph Objects for complex ones.
- Make charts visually appealing: titles, axis labels, template="plotly_dark".
- If the query asks for a "pairplot" use px.scatter_matrix instead.
- Handle missing values gracefully with dropna() where appropriate.
- If prior conversation turns are provided, use them to understand follow-up context.
- Return ONLY raw Python code — no markdown fences, no explanations."""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger.
Fix the broken code so it runs correctly.
Return ONLY the corrected Python code — no markdown, no explanations, no fences.
The fixed code must assign the figure to `fig`."""

SUMMARY_SYSTEM_PROMPT = """You are a concise, insightful data analyst.
Given a user's question and the Python/Plotly code that answered it, write a short
natural-language summary of what the chart shows and the key insights.

Rules:
- 3–5 sentences maximum.
- Lead with the most important finding.
- Mention specific numbers, trends, or comparisons where relevant.
- Plain English — no bullet points, no markdown headers.
- Describe what the DATA shows, not what the code does."""

SUGGEST_SYSTEM_PROMPT = """You are a data analyst helping a user explore a dataset.
Given a dataset schema, suggest 6 specific, interesting analytical questions the user could ask.
Each question should lead to a meaningful chart or insight.

Rules:
- Make questions specific to the actual column names and data in the schema.
- Vary chart types: include bar, scatter, line, histogram, pie, box plots.
- Keep each question under 10 words.
- Return ONLY a JSON array of 6 strings, e.g.: ["Question 1", "Question 2", ...]
- No extra text, no markdown."""

SQL_SYSTEM_PROMPT = """You are an expert SQL writer.
Given a table/database schema and a plain-English request, write a single clean SQL SELECT statement.
Return ONLY the SQL — no explanations, no markdown fences, no comments."""


class CodeGeneratorTool:
    def __init__(self, api_key: str, azure_endpoint: str, llm_deployment: str):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-02-01",
        )
        self.model = llm_deployment

    # ── Code generation (with optional conversation history) ──────────────────

    def generate(
        self,
        query: str,
        schema_context: str,
        history: list | None = None,
        stream_callback: Callable[[str], None] | None = None,
        cost_tracker=None,
    ) -> str:
        """
        Generate Python + Plotly code.

        Args:
            query:           The user's natural-language question.
            schema_context:  RAG-retrieved schema text.
            history:         List of {query, code, summary} dicts for follow-up context.
            stream_callback: Optional callable(token: str). When provided, the response
                             is streamed and each token chunk is forwarded to the callback.
            cost_tracker:    Optional CostTracker instance for recording usage.

        Returns:
            Complete Python code string.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject last N conversation turns for follow-up context
        if history:
            for turn in history[-3:]:
                messages.append({
                    "role": "user",
                    "content": f"Previous question: {turn['query']}"
                })
                messages.append({
                    "role": "assistant",
                    "content": turn["code"]
                })

        messages.append({
            "role": "user",
            "content": (
                f"Schema context:\n{schema_context}\n\n"
                f"User question: {query}\n\n"
                f"Write Python + Plotly code. df is already loaded. Assign result to `fig`."
            ),
        })

        if stream_callback is not None:
            return self._generate_streaming(messages, stream_callback, cost_tracker, operation="generate")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        if cost_tracker is not None:
            try:
                usage = response.usage
                cost_tracker.record(
                    model=self.model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    operation="generate",
                )
            except Exception:
                pass
        return self._clean_code(response.choices[0].message.content)

    def _generate_streaming(
        self,
        messages: list,
        stream_callback: Callable[[str], None],
        cost_tracker=None,
        operation: str = "generate",
    ) -> str:
        """
        Internal helper: stream tokens from Azure OpenAI and accumulate the full response.
        Falls back to a non-streaming call if the API doesn't support stream_options.
        """
        collected_tokens: list[str] = []
        token_count = 0

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except (IndexError, AttributeError):
                    delta = None
                if delta:
                    collected_tokens.append(delta)
                    token_count += 1
                    # Notify callback every 10 tokens to reduce re-render overhead
                    if token_count % 10 == 0:
                        stream_callback("".join(collected_tokens))

            # Final flush
            full_text = "".join(collected_tokens)
            stream_callback(full_text)

            # Record approximate cost (streaming doesn't provide exact usage)
            if cost_tracker is not None:
                try:
                    approx_output = max(len(full_text) // 4, 1)
                    approx_input = sum(len(m["content"]) for m in messages) // 4
                    cost_tracker.record(
                        model=self.model,
                        input_tokens=approx_input,
                        output_tokens=approx_output,
                        operation=f"{operation}_stream",
                    )
                except Exception:
                    pass

            return self._clean_code(full_text)

        except Exception:
            # Streaming failed — fall back to non-streaming
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            full_text = response.choices[0].message.content
            stream_callback(full_text)
            if cost_tracker is not None:
                try:
                    usage = response.usage
                    cost_tracker.record(
                        model=self.model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        operation=operation,
                    )
                except Exception:
                    pass
            return self._clean_code(full_text)

    # ── Self-healing fix ──────────────────────────────────────────────────────

    def fix(
        self,
        original_code: str,
        error: str,
        query: str,
        schema_context: str,
        cost_tracker=None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FIX_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Original question: {query}\n\n"
                    f"Schema:\n{schema_context}\n\n"
                    f"Broken code:\n```python\n{original_code}\n```\n\n"
                    f"Error:\n{error}\n\n"
                    f"Return only the fixed Python code."
                )},
            ],
            temperature=0.1,
        )
        if cost_tracker is not None:
            try:
                usage = response.usage
                cost_tracker.record(
                    model=self.model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    operation="fix",
                )
            except Exception:
                pass
        return self._clean_code(response.choices[0].message.content)

    # ── AI insight summary ────────────────────────────────────────────────────

    def summarize(
        self,
        query: str,
        schema_context: str,
        code: str,
        cost_tracker=None,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"User question: {query}\n\n"
                        f"Schema:\n{schema_context}\n\n"
                        f"Code that produced the chart:\n{code}\n\n"
                        f"Write a concise summary of the key insights."
                    )},
                ],
                temperature=0.4,
            )
            if cost_tracker is not None:
                try:
                    usage = response.usage
                    cost_tracker.record(
                        model=self.model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        operation="summarize",
                    )
                except Exception:
                    pass
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summary unavailable: {e}"

    # ── AI-generated query suggestions ───────────────────────────────────────

    def suggest_queries(
        self,
        schema_context: str,
        dataset_name: str,
        cost_tracker=None,
    ) -> list[str]:
        """Return 6 interesting questions tailored to the loaded dataset."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Dataset: {dataset_name}\n\n"
                        f"Schema:\n{schema_context}\n\n"
                        f"Suggest 6 specific analytical questions as a JSON array."
                    )},
                ],
                temperature=0.7,
            )
            if cost_tracker is not None:
                try:
                    usage = response.usage
                    cost_tracker.record(
                        model=self.model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        operation="suggest_queries",
                    )
                except Exception:
                    pass
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.splitlines()[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.splitlines()[:-1])
            suggestions = json.loads(raw)
            return suggestions[:6] if isinstance(suggestions, list) else []
        except Exception:
            return []

    # ── Natural language → SQL ────────────────────────────────────────────────

    def generate_sql(
        self,
        request: str,
        schema_info: str,
        cost_tracker=None,
    ) -> str:
        """Convert a plain-English data request into a SQL SELECT statement."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SQL_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Database schema:\n{schema_info}\n\n"
                        f"Request: {request}\n\n"
                        f"Write the SQL query."
                    )},
                ],
                temperature=0.1,
            )
            if cost_tracker is not None:
                try:
                    usage = response.usage
                    cost_tracker.record(
                        model=self.model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        operation="generate_sql",
                    )
                except Exception:
                    pass
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"-- SQL generation failed: {e}"

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_code(raw: str) -> str:
        lines = raw.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
