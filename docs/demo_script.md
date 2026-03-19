# 🎬 Demo Video Script — AI Analyst Agent
## Ciklum AI Academy Capstone | ~5 minutes

---

### [0:00 – 0:30] INTRO

**[Screen: README / project folder open in VS Code]**

> "Hi, I'm [Your Name] and this is my Ciklum AI Academy capstone project —
> an AI Analyst Agent that converts plain English questions into interactive
> Plotly dashboards, with a self-healing loop that automatically fixes broken code."

> "The agent combines RAG, OpenAI GPT-4o, ChromaDB, and Streamlit into a
> single pipeline. Let me walk you through how it works."

---

### [0:30 – 1:00] ARCHITECTURE OVERVIEW

**[Screen: show docs/architecture.mmd rendered, or the Streamlit sidebar]**

> "At a high level: the user types a question → the agent retrieves relevant
> schema context from a ChromaDB vector store → GPT-4o generates Python and
> Plotly code → we execute it in a sandboxed namespace → and if it fails,
> we feed the error back to GPT-4o and retry up to 3 times."

> "Finally an evaluator scores the run out of 10."

---

### [1:00 – 1:30] LOAD DATASET

**[Screen: Streamlit app, sidebar visible]**

> "I'll paste my OpenAI API key here — it's never stored."
> [paste key]

> "I'll select the Titanic dataset and click Load Dataset."
> [click button, wait for spinner]

> "The agent is chunking the dataset schema — column names, data types,
> unique values, and sample rows — and embedding them into ChromaDB.
> This is what powers our RAG retrieval."

---

### [1:30 – 2:30] HAPPY PATH — FIRST TRY SUCCESS

**[Screen: query input]**

> "Let's start with a simple query."
> [type]: "Show survival rate by passenger class as a bar chart"

> "Watch the agent log in real time — it retrieves schema context,
> generates code, and executes it."
> [wait for chart]

> "The bar chart rendered on the first attempt — score 10 out of 10."
> [point to metrics row]

> "And here's the generated code the agent wrote."
> [expand the code expander, show code briefly]

---

### [2:30 – 3:30] SELF-HEALING DEMO

**[Screen: query input]**

> "Now let me show the self-healing loop. I'll ask something more complex."
> [type]: "Plot age distribution of survivors vs non-survivors"

> [if it succeeds first try — move on]
> [if it fails and retries — narrate:]

> "You can see here — attempt 1 failed with a key error. The agent captures
> the full traceback, sends it back to GPT-4o with the original code,
> and asks it to fix the specific issue."

> "Attempt 2 — the fixed code runs successfully. The chart is rendered
> and the evaluator gives it 8 out of 10 — one point deducted because
> it needed a retry."

---

### [3:30 – 4:00] SECOND DATASET — GAPMINDER

**[Screen: switch dataset to Gapminder]**

> "Let me quickly switch to the Gapminder dataset."
> [load Gapminder]

> [type]: "Bubble chart: GDP per capita vs life expectancy sized by population"

> "This is a more complex chart — GPT-4o uses px.scatter with a size parameter."
> [wait for chart, hover over bubbles]

> "The chart is fully interactive — I can hover, zoom, and filter by continent."

---

### [4:00 – 4:30] EVALUATION + HISTORY

**[Screen: evaluation section, history table]**

> "Every run produces an evaluation report: did the code execute? did
> a chart appear? did the self-healing loop kick in?"

> "At the bottom you can see the full query history — queries, attempts,
> and scores across the session."

---

### [4:30 – 5:00] CLOSE

**[Screen: back to code / README]**

> "To summarise: this agent covers all five capstone requirements —
> data preparation and RAG, tool-calling for code generation and execution,
> self-reflection through the self-healing retry loop, and evaluation."

> "The full source code is on GitHub — link in the submission form.
> Thanks for watching!"

---

## 📝 Recording Tips
- Use OBS or Loom (free, no watermark with account)
- Resolution: 1920×1080, 30fps minimum
- Show the terminal briefly when starting `streamlit run app.py`
- Keep the browser zoom at 90–100% so the Streamlit UI is readable
- Upload to YouTube (unlisted) or Loom and paste the link into the submission form
