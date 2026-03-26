# ✦ AI Analyst Agent
**Ciklum AI Academy — Engineering Capstone Project**

An AI-powered data analyst agent that turns plain-English questions into interactive Plotly dashboards. It generates Python code, executes it in a sandboxed environment, self-heals on failure, and delivers a natural-language insight summary — all powered by **Azure OpenAI**.

---

## What It Does

1. **Ask** a question in plain English — *"Show price distribution by cut quality"*
2. **RAG retrieves** relevant schema context (column names, types, stats, sample rows) using cosine similarity over Azure OpenAI embeddings
3. **Azure GPT generates** executable Python + Plotly code tailored to your dataset
4. **Self-healing executor** runs the code in an isolated namespace — on failure, the full traceback is sent back to the model which fixes and retries (up to 3 attempts)
5. **Interactive Plotly chart** renders directly in the Streamlit app
6. **AI Insight summary** is generated — a concise 3–5 sentence natural-language explanation of what the chart reveals
7. **Evaluator** scores the run across execution, chart quality, self-healing, and code quality dimensions (0–10)
8. **Cost tracker** records every Azure OpenAI API call and displays real-time estimated USD cost

---

## Architecture

```
User Query
    │
    ▼
RAG Retrieval (Azure Embeddings + Cosine Similarity)
    │   embeds schema, column stats, sample rows → cached to disk
    ▼
Code Generator (Azure GPT)
    │   produces Python + Plotly code
    │   injects last 3 conversation turns for follow-up context
    │   streams tokens live to the UI (optional)
    ▼
Self-Healing Executor ──────────► Fail? → AI fixes code → retry (max 3×)
    │   thread-based, 30s timeout       captures full traceback each attempt
    │   isolated namespace (df, pd, np, px, go)
    │   success
    ▼
Interactive Plotly Chart
    │
    ▼
AI Insight Summary (Azure GPT)
    │   plain-English key findings (3–5 sentences)
    ▼
Evaluator (score 0–10)
    │
    ▼
Cost Tracker → records tokens + estimated USD per call
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | Azure OpenAI (GPT-5.1 + `text-embedding-3-small`) |
| RAG Vector Store | In-memory cosine similarity (NumPy) — disk-cached embeddings (pickle) |
| Agent Orchestration | Custom Python orchestrator (`AnalystAgent`) |
| Visualisation | Plotly Express / Plotly Graph Objects |
| Frontend | Streamlit — dark glassmorphism UI with light/dark theme toggle |
| Config | `python-dotenv` — credentials auto-loaded from `.env` |
| Built-in Datasets | Seaborn (Diamonds, Iris, Tips) + Plotly (Gapminder) |
| Sklearn Datasets | Diabetes, Wine, Breast Cancer, California Housing |
| External Data | CSV / Excel upload, SQL (SQLAlchemy), Microsoft Fabric |
| Cost Tracking | Custom `CostTracker` — per-call USD estimates, in-session totals |
| Evaluation | Custom `Evaluator` — multi-criterion scoring (0–10) |
| Error Telemetry | `errors.jsonl` — failed executions logged with timestamp, query, traceback |
| Query History | `history.jsonl` — successful runs persisted to disk across sessions |

---

## Project Structure

```
ai-analyst-agent/
├── app.py                    # Streamlit entry point + full UI
│                             #   auth gate, theme toggle, sidebar, chat, history, pinned charts
├── .env                      # Azure credentials (never commit this)
├── requirements.txt
│
├── agent/
│   ├── analyst_agent.py      # Main orchestrator
│   │                         #   RAG → generate → self-heal → summarize → eval → cost
│   ├── rag_context.py        # Embedding-based RAG (Azure OpenAI, cosine similarity, disk cache)
│   └── __init__.py
│
├── tools/
│   ├── code_generator.py     # Code generation, streaming, self-healing fix,
│   │                         #   insight summarization, query suggestions, NL→SQL
│   ├── executor.py           # Thread-based executor (30s timeout, error telemetry)
│   ├── cost_tracker.py       # Token usage + USD cost estimator (per-call + session total)
│   └── __init__.py
│
├── data/
│   ├── loader.py             # Dataset loaders: built-in, sklearn, CSV/Excel, SQL, Fabric
│   └── __init__.py
│
├── eval/
│   ├── evaluator.py          # Multi-criterion run scorer (0–10) + chart type detection
│   └── __init__.py
│
├── docs/
│   └── demo_script.md        # Live demo walkthrough script
│
├── history.jsonl             # Persistent query history (auto-created)
└── errors.jsonl              # Error telemetry log (auto-created)
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/supergodciklum/AI-Analyst-Agent.git
cd AI-Analyst-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure credentials

A `.example.env` file is included in the repo as a template. Copy it to `.env` and fill in your values:

```bash
cp .example.env .env
```

Then open `.env` and replace the placeholders:

```env
AZURE_OPENAI_API_KEY="your-actual-azure-api-key"
AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com"
AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
AZURE_LLM_DEPLOYMENT="gpt-5.1"
APP_PASSWORD="your-app-password"   # optional — enables login gate
```

> **Never commit `.env` to git.** It is already listed in `.gitignore`. The `.example.env` file is safe to commit — it contains only placeholder values, no real secrets.
>
> Credentials auto-load on startup and pre-fill the sidebar. You can also enter or override them directly in the UI.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Use it

1. Credentials auto-load from `.env` — a green badge confirms in the sidebar
2. Select a **data source**: built-in dataset, sklearn dataset, CSV/Excel upload, SQL connection, or Microsoft Fabric
3. Click **Load Dataset** — builds the RAG embedding index (cached to disk for subsequent loads)
4. Type a question or click one of the AI-suggested example queries
5. Watch the agent generate code, execute it, self-heal if needed, and render the chart with a live log panel

---

## Data Sources

### Built-in Datasets
| Dataset | Description |
|---|---|
| **Diamonds** | ~54,000 diamonds — carat, cut, color, clarity, price, dimensions |
| **Iris** | Classic flower dataset — sepal/petal measurements, 3 species |
| **Tips (Restaurant)** | 244 restaurant bills — tip, total, day, time, smoker, party size |
| **Gapminder** | World development 1952–2007 — GDP, life expectancy, population |

### Sklearn Datasets
| Dataset | Description |
|---|---|
| **Diabetes** | 442 patients — 10 medical features + disease progression target |
| **Wine** | 178 samples — 13 chemical features, 3 wine classes |
| **Breast Cancer** | 569 samples — 30 cell nucleus features, malignant/benign target |
| **California Housing** | ~20,000 census blocks — income, age, rooms, location, home value |

### Upload / Connect
- **CSV / Excel** — drag-and-drop file upload
- **SQL** — any SQLAlchemy connection string (PostgreSQL, MySQL, SQLite, MSSQL)
- **Microsoft Fabric** — SQL Analytics Endpoint with Entra/AAD authentication

---

## Features In Detail

### Conversation Memory
The agent retains the last 3 successful query–code–summary turns per session. Follow-up questions like *"Now colour it by clarity"* automatically use previous context without repeating the full question.

### Live Code Streaming
Code tokens stream to the UI in real time as Azure GPT generates them, with a final flush at the end. Falls back to non-streaming if the API doesn't support it.

### RAG Embedding Cache
Schema embeddings are saved to `.cache/emb_<hash>.pkl` after the first load. Reloading the same dataset skips the embedding API call entirely — the cache badge turns green in the sidebar.

### Shareable Query Links
After each query the browser URL is updated with `?q=<query>&dataset=<name>` query params. Copy the URL to share or bookmark a specific analysis.

### Pinned Charts
Successful charts can be pinned to a persistent gallery at the bottom of the page, keeping multiple analyses visible at once.

### Query Suggestions
On dataset load the agent calls Azure GPT to generate 6 dataset-specific analytical questions. Click any suggestion to instantly populate the query input.

### NL → SQL
The sidebar includes a natural-language → SQL generator. Type a plain-English request and receive a ready-to-run SELECT statement based on the loaded schema.

### Theme Toggle
Switch between dark (default glassmorphism) and light mode via the sidebar button. The theme persists for the session.

### Password Auth Gate
Set `APP_PASSWORD` in `.env` to protect the app behind a login screen. Useful for shared or deployed instances.

---

## Self-Healing Loop

```
Attempt 1 — run generated code
    ↓ fail
Capture full traceback → send to Azure GPT with original code + error
    ↓
Attempt 2 — run fixed code
    ↓ fail
Capture full traceback → send to Azure GPT again
    ↓
Attempt 3 — run fixed code
    ↓ fail → surface error to user with last traceback
```

Each retry includes the complete traceback so the model reasons about exactly what went wrong. All failures are logged to `errors.jsonl` with timestamp, dataset name, query, and a code snippet.

---

## AI Insight Summary

After a successful chart render, the agent makes a second Azure GPT call — not to write code, but to read the query and generated code and produce a **3–5 sentence natural-language summary** of the key findings:

- Leads with the most important insight
- References specific numbers, trends, and comparisons
- Displayed in a styled card directly below the chart

---

## Evaluation Scoring

| Criterion | Points |
|---|---|
| Code executed without error | 4 |
| Plotly figure produced | 2 |
| Self-healing: first try = 2 pts, retry success = 1 pt | 0–2 |
| Code quality: chart title present + non-trivial logic (≥3 lines) | 0–2 |
| **Total** | **0–10** |

Detected chart types: `bar`, `scatter`, `histogram`, `pie`, `box`, `heatmap`, `line`, `scatter_matrix`, `violin`, `bubble`.

---

## Cost Tracking

Every Azure OpenAI API call is recorded with:
- Model name, operation type (`generate`, `fix`, `summarize`, `embed`, `suggest_queries`, `generate_sql`)
- Input and output token counts
- Estimated USD cost

Pricing reference (as of 2024):

| Model | Input | Output |
|---|---|---|
| gpt-5.1 | $2.50 / 1M tokens | $10.00 / 1M tokens |
| text-embedding-3-small | $0.02 / 1M tokens | — |

The sidebar shows a live running total (`$0.0034 · 1,234 tokens`) with a Reset button.

---

## Example Queries

**Diamonds**
- Show price distribution by cut quality as a box plot
- Scatter plot of carat vs price coloured by clarity
- Bar chart of average price by color grade
- Histogram of carat distribution

**Iris**
- Scatter plot of sepal length vs petal length coloured by species
- Box plot of all features grouped by species
- Pair plot (scatter matrix) of all numeric columns

**Tips (Restaurant)**
- Bar chart of average tip by day of week
- Scatter plot of total bill vs tip coloured by smoker status
- Histogram of tip percentage distribution

**Gapminder**
- Bubble chart: GDP per capita vs life expectancy sized by population
- Line chart of life expectancy over time for India, China, and USA
- Bar chart of top 10 countries by GDP per capita in 2007

---

## Query History

All successful runs are appended to `history.jsonl` (one JSON record per line) and displayed in the **History** tab of the main panel. Each record includes the query, dataset name, score, attempts, chart type, and timestamp. The history persists across app restarts.

---

## About

Built as the capstone project for the **Ciklum AI Academy Engineering Track**.

Covers: Azure OpenAI integration · RAG pipelines · embedding caching · agentic self-reflection · LLM code generation · live token streaming · sandboxed execution · conversation memory · AI-generated insight summaries · NL→SQL · cost tracking · evaluation frameworks · multi-source data loading · Microsoft Fabric integration.

---

