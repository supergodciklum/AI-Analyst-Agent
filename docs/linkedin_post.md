# LinkedIn Post — AI Analyst Agent

---

🤖 I just built an AI Analyst Agent that turns plain English into interactive data dashboards — and fixes its own bugs.

As part of the Ciklum AI Academy Engineering Track, I designed an agentic system that:
✅ Retrieves schema context from a ChromaDB RAG vector store before generating any code
✅ Uses GPT-4o to write Python + Plotly code from a natural language query
✅ Runs a self-healing loop — if the code fails, it reads the error, fixes the code, and retries up to 3 times
✅ Scores every run with a built-in evaluator (0–10) tracking success, attempts, and chart quality

The result: ask "Show survival rate by passenger class" and get a fully interactive Plotly bar chart — no manual coding required.

This project taught me how real-world agentic systems work end to end: RAG pipelines, tool-calling, self-reflection, and evaluation — not just theory.

Huge thanks to the team @Ciklum for creating the AI Academy and making this kind of learning possible. 🙌

#AI #GenerativeAI #LLM #RAG #Python #Plotly #Streamlit #CiklumAIAcademy #MachineLearning #SoftwareEngineering
