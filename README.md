# ✦ AI Analyst Agent
**Ciklum AI Academy — Engineering Capstone Project**

An AI-powered data analyst agent that turns plain-English questions into interactive Plotly dashboards. It generates Python code, executes it in a sandboxed environment, self-heals on failure, and delivers a natural-language insight summary — all powered by **Azure OpenAI**.

---

## What It Does

1. **Ask** a question in plain English — *"Show survival rate by passenger class"*
2. **RAG retrieves** relevant schema context (column names, types, sample values) using cosine similarity over Azure OpenAI embeddings
3. **Azure GPT generates** executable Python + Plotly code tailored to your dataset
4. **Self-healing executor** runs the code — on failure, the traceback is sent back to the model which fixes and retries (up to 3 attempts)
5. **Interactive Plotly chart** renders in the Streamlit app
6. **AI Insight summary** is generated — a concise natural-language explanation of what the chart reveals
7. **Evaluator** scores the run across execution, quality, and self-healing dimensions (0–10)

---

## Architecture

```
User Query
    │
    ▼
RAG Retrieval (Azure Embeddings + Cosine Similarity)
    │   embeds schema, column stats, sample rows
    ▼
Code Generator (Azure GPT)
    │   produces Python + Plotly code
    ▼
Self-Healing Executor ──────────► Fail? → AI fixes code → retry (max 3×)
    │ success
    ▼
Interactive Plotly Chart
    │
    ▼
AI Insight Summary (Azure GPT)
    │   plain-English key findings
    ▼
Evaluator (score 0–10)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | Azure OpenAI (GPT + `text-embedding-3-small`) |
| RAG Vector Store | In-memory cosine similarity (NumPy) — no external DB |
| Agent Orchestration | Custom Python orchestrator |
| Visualisation | Plotly Express / Graph Objects |
| Frontend | Streamlit (dark glassmorphism UI) |
| Config | `python-dotenv` — credentials loaded from `.env` |
| Datasets | Seaborn + Plotly built-ins (Titanic, Iris, Tips, Gapminder) |

---

## Project Structure

```
ai-analyst-agent/
├── app.py                  # Streamlit entry point + UI
├── .env                    # Azure credentials (never commit this)
├── requirements.txt
├── agent/
│   ├── analyst_agent.py    # Main orchestrator (RAG → generate → heal → summarize → eval)
│   ├── rag_context.py      # Embedding-based RAG index (Azure OpenAI, cosine similarity)
│   └── __init__.py
├── tools/
│   ├── code_generator.py   # Code generation, self-healing fix, and insight summarization
│   └── executor.py         # Sandboxed Python executor
├── data/
│   └── loader.py           # Public dataset loader
├── eval/
│   └── evaluator.py        # Run scorer (0–10)
└── docs/
    └── architecture.mmd    # Mermaid architecture diagram
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-analyst-agent.git
cd ai-analyst-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure credentials

Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY="your-azure-api-key"
AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com"
AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
AZURE_LLM_DEPLOYMENT="gpt-4o"
```

> Credentials are loaded automatically on startup via `python-dotenv` and pre-fill the sidebar fields. You can also enter or override them directly in the UI.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Use it

1. Credentials auto-load from `.env` — confirm the green badge in the sidebar
2. Select a **dataset** (Titanic / Iris / Tips / Gapminder)
3. Click **Load Dataset** — builds the RAG embedding index
4. Type a question or click an example prompt
5. Watch the agent think, generate, execute, self-heal, and summarize in real time

---

## Self-Healing Loop

```
Attempt 1 — run generated code
    ↓ fail
Capture traceback → send to Azure GPT with original code + error
    ↓
Attempt 2 — run fixed code
    ↓ fail
Capture traceback → send to Azure GPT again
    ↓
Attempt 3 — run fixed code
    ↓ fail → surface error to user
```

Each retry includes the full traceback so the model reasons about exactly what went wrong rather than guessing.

---

## AI Insight Summary

After a successful chart render, the agent calls Azure GPT a second time — not to write code, but to read the query and the generated code and produce a **3–5 sentence natural-language summary** of the key findings:

- Leads with the most important insight
- References specific numbers, trends, and comparisons from the data
- Displayed in a styled card directly below the chart

---

## Evaluation Scoring

| Criterion | Points |
|---|---|
| Code executed without error | 4 |
| Plotly figure produced | 2 |
| Self-healing (first try = 2 pts, retry success = 1 pt) | 0–2 |
| Code quality (chart title + non-trivial logic) | 0–2 |
| **Total** | **0–10** |

---

## Example Queries

**Titanic**
- Show survival rate by passenger class as a bar chart
- Plot age distribution of survivors vs non-survivors
- Heatmap of correlations between numeric columns

**Iris**
- Scatter plot of sepal length vs petal length coloured by species
- Box plot of all features grouped by species

**Tips (Restaurant)**
- Bar chart of average tip by day of week
- Scatter plot of total bill vs tip coloured by smoker

**Gapminder**
- Bubble chart: GDP per capita vs life expectancy sized by population
- Line chart of life expectancy over time for India, China, and USA

---

## About

Built as the capstone project for the **Ciklum AI Academy Engineering Track**.

Covers: Azure OpenAI integration · RAG pipelines · agentic self-reflection · LLM code generation · sandboxed execution · AI-generated insight summaries · evaluation frameworks.

---

## License

MIT
