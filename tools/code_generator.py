"""
Code generator tool.
Takes a user query + RAG schema context → returns executable Python + Plotly code.
Uses Azure OpenAI.
"""

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
- Make charts visually appealing: add titles, axis labels, and use a clean template like "plotly_dark".
- If the query asks for a "pairplot" use px.scatter_matrix instead (seaborn pairplot not available).
- Handle missing values gracefully with dropna() where appropriate.
- Keep the code concise and correct. No explanations, no markdown, just raw Python code.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. You will receive Python code
that failed with an error, along with the full error traceback.
Fix the code so it runs correctly. Return ONLY the corrected Python code — no markdown,
no explanations, no code fences. The fixed code must assign the figure to `fig`."""

SUMMARY_SYSTEM_PROMPT = """You are a concise, insightful data analyst.
Given a user's question and the Python/Plotly code that answered it, write a short
natural-language summary of what the chart shows and the key insights.

Rules:
- 3–5 sentences maximum.
- Lead with the most important finding.
- Mention specific numbers, trends, or comparisons where relevant.
- Write in plain English — no jargon, no bullet points, no markdown headers.
- Do NOT describe what the code does — describe what the DATA shows."""


class CodeGeneratorTool:
    def __init__(self, api_key: str, azure_endpoint: str, llm_deployment: str):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-02-01",
        )
        self.model = llm_deployment

    def generate(self, query: str, schema_context: str) -> str:
        """Generate Python + Plotly code from a natural language query."""
        user_message = f"""Schema context:
{schema_context}

User question: {query}

Write Python code using Pandas + Plotly that answers this question.
Remember: df is already loaded. Assign the result to `fig`."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        return self._clean_code(response.choices[0].message.content)

    def fix(self, original_code: str, error: str, query: str, schema_context: str) -> str:
        """Given broken code + error traceback, return fixed code."""
        user_message = f"""Original user question: {query}

Schema context:
{schema_context}

Broken code:
```python
{original_code}
```

Error traceback:
{error}

Fix the code. Return only corrected Python — no markdown, no fences."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FIX_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        return self._clean_code(response.choices[0].message.content)

    def summarize(self, query: str, schema_context: str, code: str) -> str:
        """Generate a plain-English summary of what the chart reveals."""
        user_message = f"""User question: {query}

Schema context:
{schema_context}

Python code that produced the chart:
{code}

Write a concise natural-language summary of the key insights this chart reveals."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summary unavailable: {e}"

    @staticmethod
    def _clean_code(raw: str) -> str:
        """Strip markdown code fences if the model added them."""
        lines = raw.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
