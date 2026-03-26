"""
AnalystAgent — the main orchestrator.

Flow per query:
  1. RAG retrieval → schema context
  2. CodeGeneratorTool.generate() with conversation history → Python + Plotly code
  3. SelfHealingExecutor → run code
  4. On failure → CodeGeneratorTool.fix() → retry (max 3 times)
  5. AI insight summary
  6. Return full result dict

Features:
- Optional stream_callback for live code token streaming in the UI
- Optional cost_tracker for recording API usage
- Passes meta (dataset, query) to executor for error telemetry
"""

from typing import Callable, Optional

import pandas as pd
from agent.rag_context import RAGContext
from tools.code_generator import CodeGeneratorTool
from tools.executor import SelfHealingExecutor, MAX_ATTEMPTS


class AnalystAgent:
    def __init__(
        self,
        api_key: str,
        df: pd.DataFrame,
        dataset_name: str,
        dataset_description: str,
        azure_endpoint: str,
        embedding_deployment: str,
        llm_deployment: str,
        cost_tracker=None,
    ):
        self.df = df
        self.dataset_name = dataset_name
        self.cost_tracker = cost_tracker

        self.rag = RAGContext(
            api_key=api_key,
            df=df,
            dataset_name=dataset_name,
            description=dataset_description,
            azure_endpoint=azure_endpoint,
            embedding_deployment=embedding_deployment,
            cost_tracker=cost_tracker,
        )
        self.generator = CodeGeneratorTool(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            llm_deployment=llm_deployment,
        )
        self.executor = SelfHealingExecutor(df=df)

        # Conversation memory — last N successful turns
        self.conversation_history: list[dict] = []

        # Expose cache flag for UI feedback
        self.rag_from_cache: bool = self.rag._from_cache

    # ── Query suggestions (called once after load) ────────────────────────────

    def suggest_queries(self) -> list[str]:
        schema = self.rag.full_schema_summary()
        return self.generator.suggest_queries(
            schema,
            self.rag.dataset_name,
            cost_tracker=self.cost_tracker,
        )

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        log_callback: Callable | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> dict:
        def log(type_, msg):
            if log_callback:
                log_callback(type_, msg)

        meta = {"dataset": self.dataset_name, "query": query}

        # Step 1: RAG retrieval
        log("", "🔎 Retrieving schema context from vector store…")
        try:
            context = self.rag.retrieve(query, n_results=5)
            log("success", "✅ Schema context retrieved.")
        except Exception as e:
            log("error", f"⚠️ RAG retrieval failed, using full schema. ({e})")
            context = self.rag.full_schema_summary()

        # Step 2: Code generation (pass conversation history for follow-up support)
        log("", "🧠 Generating Python + Plotly code…")
        code = self.generator.generate(
            query=query,
            schema_context=context,
            history=self.conversation_history,
            stream_callback=stream_callback,
            cost_tracker=self.cost_tracker,
        )
        log("", f"📝 Code generated ({len(code.splitlines())} lines).")

        last_error = None
        attempt = 0

        # Step 3: Self-healing execution loop
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            if attempt == 1:
                log("", f"⚙️ Attempt {attempt}/{MAX_ATTEMPTS} — executing code…")
            else:
                log("retry", f"🔄 Attempt {attempt}/{MAX_ATTEMPTS} — retrying with fixed code…")

            result = self.executor.execute(code, log_callback=log_callback, meta=meta)

            if result["success"]:
                log("success", f"✅ Execution succeeded on attempt {attempt}.")
                log("", "📝 Generating insight summary…")
                summary = self.generator.summarize(
                    query=query,
                    schema_context=context,
                    code=code,
                    cost_tracker=self.cost_tracker,
                )

                # Save to conversation history for future follow-ups
                self.conversation_history.append({
                    "query": query,
                    "code": code,
                    "summary": summary,
                })

                return {
                    "success": True,
                    "fig": result["fig"],
                    "code": code,
                    "attempts": attempt,
                    "last_error": None,
                    "summary": summary,
                }

            last_error = result["error"]
            if attempt < MAX_ATTEMPTS:
                log("retry", f"🔧 Asking AI to fix the error… ({attempt}/{MAX_ATTEMPTS - 1} fix attempts)")
                code = self.generator.fix(
                    original_code=code,
                    error=last_error,
                    query=query,
                    schema_context=context,
                    cost_tracker=self.cost_tracker,
                )
                log("", f"📝 Fixed code generated ({len(code.splitlines())} lines).")

        log("error", f"❌ All {MAX_ATTEMPTS} attempts failed.")
        return {
            "success": False,
            "fig": None,
            "code": code,
            "attempts": attempt,
            "last_error": last_error,
            "summary": "",
        }

    def clear_history(self):
        self.conversation_history = []
