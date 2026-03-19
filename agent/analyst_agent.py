"""
AnalystAgent — the main orchestrator.

Flow per query:
  1. RAG retrieval → schema context
  2. CodeGeneratorTool → Python + Plotly code
  3. SelfHealingExecutor → run code
  4. On failure → CodeGeneratorTool.fix() → retry (max 3 times)
  5. Return result dict with fig, code, attempts, success
"""

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
    ):
        self.api_key = api_key
        self.df = df

        # Build RAG index
        self.rag = RAGContext(
            api_key=api_key,
            df=df,
            dataset_name=dataset_name,
            description=dataset_description,
            azure_endpoint=azure_endpoint,
            embedding_deployment=embedding_deployment,
        )
        self.generator = CodeGeneratorTool(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            llm_deployment=llm_deployment,
        )
        self.executor = SelfHealingExecutor(df=df)

    def run(self, query: str, log_callback=None) -> dict:
        """
        Execute a full agent run for a user query.
        Returns: {
            "success": bool,
            "fig": plotly figure or None,
            "code": last attempted code,
            "attempts": int,
            "last_error": str or None,
        }
        """
        def log(type_, msg):
            if log_callback:
                log_callback(type_, msg)

        # ── Step 1: RAG retrieval ──────────────────────────────────────────
        log("", "🔎 Retrieving schema context from vector store…")
        try:
            context = self.rag.retrieve(query, n_results=5)
            log("success", "✅ Schema context retrieved.")
        except Exception as e:
            log("error", f"⚠️ RAG retrieval failed, using full schema. ({e})")
            context = self.rag.full_schema_summary()

        # ── Step 2: Initial code generation ───────────────────────────────
        log("", "🧠 Generating Python + Plotly code…")
        code = self.generator.generate(query=query, schema_context=context)
        log("", f"📝 Code generated ({len(code.splitlines())} lines).")

        last_error = None
        attempt = 0

        # ── Step 3: Self-healing execution loop ───────────────────────────
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            if attempt == 1:
                log("", f"⚙️ Attempt {attempt}/{MAX_ATTEMPTS} — executing code…")
            else:
                log("retry", f"🔄 Attempt {attempt}/{MAX_ATTEMPTS} — retrying with fixed code…")

            result = self.executor.execute(code, log_callback=log_callback)

            if result["success"]:
                log("success", f"✅ Execution succeeded on attempt {attempt}.")
                log("", "📝 Generating insight summary…")
                summary = self.generator.summarize(
                    query=query, schema_context=context, code=code
                )
                return {
                    "success": True,
                    "fig": result["fig"],
                    "code": code,
                    "attempts": attempt,
                    "last_error": None,
                    "summary": summary,
                }

            # Execution failed — capture error and ask for a fix
            last_error = result["error"]
            if attempt < MAX_ATTEMPTS:
                log("retry", f"🔧 Asking AI to fix the error… ({attempt}/{MAX_ATTEMPTS - 1} fix attempts)")
                code = self.generator.fix(
                    original_code=code,
                    error=last_error,
                    query=query,
                    schema_context=context,
                )
                log("", f"📝 Fixed code generated ({len(code.splitlines())} lines).")

        # All attempts exhausted
        log("error", f"❌ All {MAX_ATTEMPTS} attempts failed. Last error captured.")
        return {
            "success": False,
            "fig": None,
            "code": code,
            "attempts": attempt,
            "last_error": last_error,
        }
