"""
AI Analyst Agent — Streamlit App
Ciklum AI Academy Capstone Project

Flow: User query → RAG context → Code generation → Self-healing execution → Plotly dashboard
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from agent.analyst_agent import AnalystAgent
from data.loader import load_dataset
from eval.evaluator import Evaluator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analyst Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #070b14;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #080c18 100%);
    border-right: 1px solid rgba(124, 106, 255, 0.15);
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,106,255,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 20px;
}
.sidebar-brand-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #7c6aff, #a855f7);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; line-height: 1;
}
.sidebar-brand-text { line-height: 1.2; }
.sidebar-brand-title {
    font-size: 15px; font-weight: 700; color: #f1f5f9;
}
.sidebar-brand-sub {
    font-size: 11px; color: #64748b;
}

/* ── Section headers inside sidebar ── */
.sidebar-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 20px 0 10px;
    padding-left: 2px;
}

/* ── Credential badge ── */
.cred-badge {
    display: flex; align-items: center; gap: 8px;
    background: rgba(74, 222, 128, 0.08);
    border: 1px solid rgba(74, 222, 128, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 8px 0 0;
    font-size: 12px;
    color: #86efac;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg,
        rgba(124,106,255,0.12) 0%,
        rgba(168,85,247,0.08) 50%,
        rgba(6,182,212,0.06) 100%);
    border: 1px solid rgba(124,106,255,0.2);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(124,106,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(124,106,255,0.15);
    border: 1px solid rgba(124,106,255,0.3);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 11px; font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 32px; font-weight: 700;
    background: linear-gradient(135deg, #f1f5f9 0%, #a78bfa 60%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 10px; line-height: 1.2;
}
.hero-subtitle {
    font-size: 15px; color: #64748b; margin: 0; max-width: 560px; line-height: 1.6;
}
.hero-stats {
    display: flex; gap: 24px; margin-top: 24px; flex-wrap: wrap;
}
.hero-stat {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; color: #475569;
}
.hero-stat-dot {
    width: 6px; height: 6px;
    background: linear-gradient(135deg, #7c6aff, #a855f7);
    border-radius: 50%;
}

/* ── Section title ── */
.section-title {
    font-size: 13px; font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,0.2), transparent);
}

/* ── Example prompt cards ── */
.prompt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 24px; }
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(124,106,255,0.2) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 10px 14px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    background: rgba(124,106,255,0.1) !important;
    border-color: rgba(124,106,255,0.5) !important;
    color: #c4b5fd !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.15) !important;
}
/* sidebar Load Dataset button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #7c6aff, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,106,255,0.35) !important;
    color: white !important;
}

/* ── Agent log panel ── */
.log-panel {
    background: rgba(13,17,32,0.8);
    border: 1px solid rgba(124,106,255,0.15);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 20px;
}
.log-header {
    font-size: 11px; font-weight: 600;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.log-entry {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 10px;
    border-radius: 8px;
    margin: 4px 0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 12.5px;
    line-height: 1.5;
    color: #94a3b8;
    background: transparent;
    border-left: 2px solid transparent;
    transition: background 0.15s;
}
.log-entry.success {
    border-left-color: #4ade80;
    color: #86efac;
    background: rgba(74,222,128,0.04);
}
.log-entry.error {
    border-left-color: #f87171;
    color: #fca5a5;
    background: rgba(248,113,113,0.04);
}
.log-entry.retry {
    border-left-color: #fbbf24;
    color: #fde68a;
    background: rgba(251,191,36,0.04);
}

/* ── Metric cards ── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 20px 0; }
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #7c6aff, #a855f7, #06b6d4);
}
.metric-label {
    font-size: 11px; font-weight: 600;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 22px; font-weight: 700;
    color: #f1f5f9;
}
.metric-value.success-val { color: #4ade80; }
.metric-value.fail-val    { color: #f87171; }

/* ── Schema pill ── */
.schema-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px; color: #64748b;
    margin: 2px;
}
.schema-pill.num { border-color: rgba(124,106,255,0.25); color: #a78bfa; }
.schema-pill.cat { border-color: rgba(6,182,212,0.25); color: #67e8f9; }

/* ── Dataset info card ── */
.dataset-card {
    background: rgba(124,106,255,0.06);
    border: 1px solid rgba(124,106,255,0.2);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 12px;
}
.dataset-card-title {
    font-size: 13px; font-weight: 600; color: #c4b5fd; margin-bottom: 4px;
}
.dataset-card-meta { font-size: 12px; color: #64748b; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,106,255,0.25) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(124,106,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,106,255,0.1) !important;
}

/* ── Query header ── */
.query-header {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 20px 0 16px;
    display: flex; align-items: center; gap: 12px;
}
.query-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #7c6aff22, #a855f722);
    border: 1px solid rgba(124,106,255,0.3);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.query-text {
    font-size: 16px; font-weight: 500; color: #e2e8f0;
    font-style: italic;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Code block ── */
.stCodeBlock { border-radius: 12px !important; border: 1px solid rgba(124,106,255,0.15) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Alert / error ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 32px 0 8px;
    font-size: 12px;
    color: #1e293b;
}
.app-footer span { color: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Azure credentials from .env ───────────────────────────────────────────────
ENV_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY", "")
ENV_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
ENV_EMBEDDING = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
ENV_LLM       = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("agent", None), ("history", []),
    ("dataset_name", None), ("df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">✦</div>
        <div class="sidebar-brand-text">
            <div class="sidebar-brand-title">AI Analyst Agent</div>
            <div class="sidebar-brand-sub">Ciklum AI Academy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Azure credentials ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🔑 Azure Credentials</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "API Key", value=ENV_API_KEY, type="password",
        placeholder="Your Azure OpenAI key…",
    )
    azure_endpoint = st.text_input(
        "Endpoint", value=ENV_ENDPOINT,
        placeholder="https://your-resource.cognitiveservices.azure.com",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        embedding_deployment = st.text_input("Embedding Model", value=ENV_EMBEDDING)
    with col_b:
        llm_deployment = st.text_input("LLM Deployment", value=ENV_LLM)

    if ENV_API_KEY:
        st.markdown('<div class="cred-badge">✅ &nbsp;Loaded from .env</div>', unsafe_allow_html=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📦 Dataset</div>', unsafe_allow_html=True)

    dataset_choice = st.selectbox(
        "Select dataset",
        ["Titanic", "Iris", "Tips (Restaurant)", "Gapminder"],
        label_visibility="collapsed",
    )

    load_clicked = st.button("⚡  Load Dataset", use_container_width=True)

    if load_clicked:
        if not api_key:
            st.error("API key is required.")
        elif not azure_endpoint:
            st.error("Azure endpoint is required.")
        else:
            with st.spinner("Building RAG index…"):
                try:
                    df, description = load_dataset(dataset_choice)
                    st.session_state.df = df
                    st.session_state.dataset_name = dataset_choice
                    st.session_state.agent = AnalystAgent(
                        api_key=api_key, df=df,
                        dataset_name=dataset_choice,
                        dataset_description=description,
                        azure_endpoint=azure_endpoint,
                        embedding_deployment=embedding_deployment,
                        llm_deployment=llm_deployment,
                    )
                    st.success(f"Loaded · {len(df):,} rows · {len(df.columns)} cols")
                except Exception as e:
                    st.error(f"{e}")

    # Dataset info + schema
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        <div class="dataset-card">
            <div class="dataset-card-title">{st.session_state.dataset_name}</div>
            <div class="dataset-card-meta">{len(df):,} rows &nbsp;·&nbsp; {len(df.columns)} columns</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">📋 Schema</div>', unsafe_allow_html=True)
        for col in df.columns:
            dtype = str(df[col].dtype)
            is_num = "int" in dtype or "float" in dtype
            css   = "num" if is_num else "cat"
            icon  = "⬡" if is_num else "⬠"
            st.markdown(
                f'<span class="schema-pill {css}">{icon} {col} <span style="opacity:.5">{dtype}</span></span>',
                unsafe_allow_html=True,
            )

    # Footer
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#1e293b;text-align:center;padding-top:8px;">'
        'Azure OpenAI · Plotly · Streamlit</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ &nbsp;AI-Powered Analytics</div>
    <div class="hero-title">Turn Questions into Insights</div>
    <p class="hero-subtitle">
        Ask a plain-English question about your data — the agent writes, executes,
        and self-heals Python + Plotly code to produce a stunning interactive chart.
    </p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-dot"></div>Azure OpenAI · GPT</div>
        <div class="hero-stat"><div class="hero-stat-dot"></div>Self-Healing Execution</div>
        <div class="hero-stat"><div class="hero-stat-dot"></div>RAG Schema Context</div>
        <div class="hero-stat"><div class="hero-stat-dot"></div>Interactive Plotly Charts</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Gate ─────────────────────────────────────────────────────────────────────
if st.session_state.agent is None:
    st.markdown("""
    <div style="
        text-align:center; padding:60px 20px;
        background:rgba(255,255,255,0.02);
        border:1px dashed rgba(124,106,255,0.2);
        border-radius:16px; color:#334155;
    ">
        <div style="font-size:40px;margin-bottom:12px;">📦</div>
        <div style="font-size:16px;font-weight:600;color:#475569;margin-bottom:6px;">No dataset loaded yet</div>
        <div style="font-size:13px;">Open the sidebar, enter your credentials and click <b>Load Dataset</b> to begin.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Example prompts ───────────────────────────────────────────────────────────
examples = {
    "Titanic": [
        "Survival rate by passenger class — bar chart",
        "Age distribution: survivors vs non-survivors",
        "Correlation heatmap of numeric columns",
        "Pie chart of embarkation ports",
    ],
    "Iris": [
        "Sepal length vs petal length by species",
        "Box plot of all features by species",
        "Pairplot (scatter matrix) of all features",
        "Histogram of petal width per species",
    ],
    "Tips (Restaurant)": [
        "Average tip by day of week",
        "Total bill vs tip coloured by smoker",
        "Box plot of tips by gender",
        "Tip percentage distribution histogram",
    ],
    "Gapminder": [
        "GDP per capita vs life expectancy scatter",
        "Top 10 countries by population bar chart",
        "Life expectancy over time for 5 countries",
        "Bubble chart: GDP vs life expectancy × population",
    ],
}

example_list = examples.get(st.session_state.dataset_name, [])
if example_list:
    st.markdown('<div class="section-title">💡 Example Prompts</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(example_list):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.prefill = ex

# ── Chat input ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)
query = st.chat_input("e.g. Show survival rate by passenger class as a bar chart…")
if "prefill" in st.session_state:
    query = st.session_state.pop("prefill")

# ── Agent run ─────────────────────────────────────────────────────────────────
if query:
    st.markdown(f"""
    <div class="query-header">
        <div class="query-icon">🔍</div>
        <div class="query-text">{query}</div>
    </div>
    """, unsafe_allow_html=True)

    log_placeholder   = st.empty()
    chart_placeholder = st.empty()

    logs = []

    def update_logs():
        entries = "\n".join(
            f'<div class="log-entry {l["type"]}">{l["msg"]}</div>'
            for l in logs
        )
        log_placeholder.markdown(
            f'<div class="log-panel">'
            f'<div class="log-header">◎ &nbsp;Agent Log</div>'
            f'{entries}'
            f'</div>',
            unsafe_allow_html=True,
        )

    logs.append({"type": "", "msg": "🔎 Retrieving schema context from RAG index…"})
    update_logs()

    result = st.session_state.agent.run(query, log_callback=lambda t, m: (
        logs.append({"type": t, "msg": m}), update_logs()
    ))

    evaluator  = Evaluator()
    eval_result = evaluator.evaluate(
        query=query,
        code=result.get("code", ""),
        success=result.get("success", False),
        attempts=result.get("attempts", 1),
        fig=result.get("fig"),
    )

    if result.get("success") and result.get("fig"):
        logs.append({"type": "success", "msg": "✅ Chart rendered successfully!"})
        update_logs()
        chart_placeholder.plotly_chart(result["fig"], use_container_width=True)

        # ── AI Summary ────────────────────────────────────────────────────────
        summary = result.get("summary", "")
        if summary:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(124,106,255,0.07), rgba(6,182,212,0.05));
                border: 1px solid rgba(124,106,255,0.2);
                border-radius: 14px;
                padding: 20px 24px;
                margin: 4px 0 20px;
                position: relative;
            ">
                <div style="
                    font-size: 10px; font-weight: 700;
                    letter-spacing: 0.12em; text-transform: uppercase;
                    color: #7c6aff; margin-bottom: 10px;
                    display: flex; align-items: center; gap: 6px;
                ">
                    ✦ &nbsp;AI Insight
                </div>
                <div style="
                    font-size: 14.5px; line-height: 1.75;
                    color: #cbd5e1; font-weight: 400;
                ">
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        logs.append({"type": "error", "msg": f"❌ Failed after {result.get('attempts', 3)} attempts."})
        update_logs()
        if result.get("last_error"):
            st.error(f"**Last error:** {result['last_error']}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    success      = result["success"]
    attempts     = result.get("attempts", 1)
    score        = eval_result["score"]
    self_healed  = attempts > 1 and success

    status_val   = "✅ Success" if success else "❌ Failed"
    status_css   = "success-val" if success else "fail-val"
    heal_val     = "✦ Yes" if self_healed else "— No"
    heal_css     = "success-val" if self_healed else ""
    score_css    = "success-val" if score >= 7 else ("fail-val" if score < 5 else "")

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value {status_css}">{status_val}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Attempts</div>
            <div class="metric-value">{attempts} <span style="font-size:14px;color:#475569">/ 3</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Eval Score</div>
            <div class="metric-value {score_css}">{score} <span style="font-size:14px;color:#475569">/ 10</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Self-Healed</div>
            <div class="metric-value {heal_css}">{heal_val}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Code + eval expanders ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if result.get("code"):
            with st.expander("🧾 Generated Python Code"):
                st.code(result["code"], language="python")
    with c2:
        with st.expander("📈 Evaluation Breakdown"):
            st.json(eval_result)

    # ── History ───────────────────────────────────────────────────────────────
    st.session_state.history.append({
        "Query": query,
        "Status": "✅" if result["success"] else "❌",
        "Attempts": attempts,
        "Score": f"{score}/10",
        "Self-healed": "Yes" if self_healed else "No",
        "Chart Type": eval_result.get("chart_type_detected", "—"),
    })

# ── History table ─────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📜 Session History</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <span>Built with</span> Azure OpenAI · ChromaDB RAG · Plotly · Streamlit
    &nbsp;·&nbsp; Ciklum AI Academy Capstone
</div>
""", unsafe_allow_html=True)
