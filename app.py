"""
AI Analyst Agent — Streamlit App
Ciklum AI Academy Capstone Project
"""

import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from agent.analyst_agent import AnalystAgent
from data.loader import load_builtin, load_sklearn, load_file, load_from_sql, load_from_fabric
from eval.evaluator import Evaluator
from tools.cost_tracker import CostTracker

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Analyst Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Persistent history helpers ────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.jsonl")
ERRORS_LOG   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "errors.jsonl")

def _load_history_file() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def _append_history(entry: dict):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def _load_errors(n: int = 20) -> list:
    if not os.path.exists(ERRORS_LOG):
        return []
    rows = []
    with open(ERRORS_LOG) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows[-n:]

# ── Authentication ────────────────────────────────────────────────────────────
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "agent": None, "df": None, "dataset_name": None,
    "history": _load_history_file(),
    "suggested_queries": [],
    "sql_generated": "",
    "authenticated": False,
    "onboarding_done": False,
    "pinned_charts": [],
    "cost_tracker": CostTracker(),
    "theme": "dark",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Shareable Query Links — read query params on load ─────────────────────────
_params = st.query_params
if "dataset" in _params and "prefill_dataset" not in st.session_state:
    st.session_state.prefill_dataset = _params["dataset"]
if "q" in _params and "prefill_query" not in st.session_state:
    st.session_state.prefill_query = _params["q"]

# ── Authentication Gate ───────────────────────────────────────────────────────
if APP_PASSWORD and not st.session_state.authenticated:
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px; margin: 80px auto; text-align: center;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(124,106,255,0.2);
        border-radius: 20px; padding: 48px 40px;
    }
    .auth-logo {
        width: 56px; height: 56px;
        background: linear-gradient(135deg, #7c6aff, #a855f7);
        border-radius: 14px; display: inline-flex; align-items: center;
        justify-content: center; font-size: 28px; margin-bottom: 20px;
    }
    .auth-title { font-size: 22px; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; }
    .auth-sub   { font-size: 13px; color: #64748b; margin-bottom: 28px; }
    </style>
    <div class="auth-container">
        <div class="auth-logo">✦</div>
        <div class="auth-title">AI Analyst Agent</div>
        <div class="auth-sub">Enter the application password to continue.</div>
    </div>
    """, unsafe_allow_html=True)

    pw_input = st.text_input("Password", type="password", placeholder="Enter password…",
                              label_visibility="collapsed")
    if st.button("Login", use_container_width=True):
        if pw_input == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ── Theme CSS injection ───────────────────────────────────────────────────────
_theme = st.session_state.theme

if _theme == "light":
    THEME_CSS = """
    .stApp { background: #f8fafc !important; }
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid rgba(124,106,255,0.15) !important;
    }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(0,0,0,0.02) !important;
        border: 1px solid rgba(124,106,255,0.25) !important;
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] label { color: #475569 !important; }
    .sidebar-brand-title { color: #1e293b !important; }
    .sidebar-brand-sub   { color: #94a3b8 !important; }
    .hero { background: linear-gradient(135deg, rgba(124,106,255,0.06) 0%, rgba(168,85,247,0.04) 50%, rgba(6,182,212,0.03) 100%) !important; border-color: rgba(124,106,255,0.15) !important; }
    .hero-subtitle { color: #475569 !important; }
    .hero-stat     { color: #64748b !important; }
    .metric-card { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.06) !important; }
    .metric-label { color: #64748b !important; }
    .metric-value { color: #1e293b !important; }
    .log-panel { background: rgba(248,250,252,0.9) !important; border-color: rgba(124,106,255,0.12) !important; }
    .log-entry { color: #475569 !important; }
    .log-entry.success { background: rgba(74,222,128,0.06) !important; }
    .log-entry.error   { background: rgba(248,113,113,0.06) !important; }
    .profiler-card { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.06) !important; }
    .profiler-col-name { color: #1e293b !important; }
    .profiler-dtype    { color: #64748b !important; }
    .profiler-stat     { color: #475569 !important; }
    .profiler-stat span { color: #334155 !important; }
    .conv-turn { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.06) !important; }
    .conv-q { color: #7c6aff !important; }
    .conv-s { color: #475569 !important; }
    .query-header { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.06) !important; }
    .query-text { color: #1e293b !important; }
    .dataset-card { background: rgba(124,106,255,0.04) !important; }
    .dataset-card-title { color: #7c6aff !important; }
    .dataset-card-meta  { color: #64748b !important; }
    .section-title { color: #64748b !important; }
    hr { border-color: rgba(0,0,0,0.06) !important; }
    [data-testid="stDataFrame"] { border-color: rgba(0,0,0,0.06) !important; }
    .stCodeBlock { border-color: rgba(124,106,255,0.12) !important; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.06) !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b !important; }
    """
else:
    THEME_CSS = ""

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: #070b14; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1120 0%, #080c18 100%);
    border-right: 1px solid rgba(124,106,255,0.15);
}}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stTextArea textarea {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,106,255,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}}
[data-testid="stSidebar"] label {{
    color: #94a3b8 !important; font-size: 12px !important;
    font-weight: 500 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}}

/* ── Radio source selector ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ display:flex; flex-direction:column; gap:4px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important; padding: 8px 12px !important;
    font-size: 12px !important; color: #64748b !important;
    cursor: pointer !important; transition: all 0.15s !important;
    text-transform: none !important; letter-spacing: normal !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
    background: rgba(124,106,255,0.12) !important;
    border-color: rgba(124,106,255,0.4) !important; color: #c4b5fd !important;
}}

/* ── File uploader ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(124,106,255,0.3) !important;
    border-radius: 10px !important; padding: 8px !important;
}}

/* ── Sidebar brand ── */
.sidebar-brand {{ display:flex; align-items:center; gap:12px; padding:8px 0 20px; }}
.sidebar-brand-icon {{
    width:40px; height:40px;
    background: linear-gradient(135deg, #7c6aff, #a855f7);
    border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:20px; line-height:1;
}}
.sidebar-brand-title {{ font-size:15px; font-weight:700; color:#f1f5f9; }}
.sidebar-brand-sub   {{ font-size:11px; color:#64748b; }}
.sidebar-section {{
    font-size:10px; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; color:#475569;
    margin:20px 0 10px; padding-left:2px;
}}
.cred-badge {{
    display:flex; align-items:center; gap:8px;
    background:rgba(74,222,128,0.08);
    border:1px solid rgba(74,222,128,0.2);
    border-radius:8px; padding:8px 12px; margin:8px 0 0;
    font-size:12px; color:#86efac;
}}
.cache-badge {{
    display:flex; align-items:center; gap:8px;
    background:rgba(6,182,212,0.08);
    border:1px solid rgba(6,182,212,0.2);
    border-radius:8px; padding:6px 12px; margin:6px 0 0;
    font-size:11px; color:#67e8f9;
}}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {{
    background: linear-gradient(135deg, #7c6aff, #a855f7) !important;
    border: none !important; color: white !important;
    font-weight: 600 !important; border-radius: 10px !important; padding: 12px !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    opacity: 0.9 !important; transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,106,255,0.35) !important; color: white !important;
}}

/* ── Hero ── */
.hero {{
    background: linear-gradient(135deg,
        rgba(124,106,255,0.12) 0%, rgba(168,85,247,0.08) 50%, rgba(6,182,212,0.06) 100%);
    border: 1px solid rgba(124,106,255,0.2);
    border-radius: 20px; padding: 36px 40px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}}
.hero::before {{
    content:''; position:absolute; top:-60px; right:-60px;
    width:240px; height:240px;
    background:radial-gradient(circle, rgba(124,106,255,0.15) 0%, transparent 70%);
    border-radius:50%;
}}
.hero-badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(124,106,255,0.15); border:1px solid rgba(124,106,255,0.3);
    border-radius:999px; padding:4px 12px; font-size:11px; font-weight:600;
    color:#a78bfa; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:14px;
}}
.hero-title {{
    font-size:32px; font-weight:700;
    background:linear-gradient(135deg, #f1f5f9 0%, #a78bfa 60%, #67e8f9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 10px; line-height:1.2;
}}
.hero-subtitle {{ font-size:15px; color:#64748b; margin:0; max-width:560px; line-height:1.6; }}
.hero-stats {{ display:flex; gap:24px; margin-top:24px; flex-wrap:wrap; }}
.hero-stat  {{ display:flex; align-items:center; gap:8px; font-size:13px; color:#475569; }}
.hero-stat-dot {{
    width:6px; height:6px;
    background:linear-gradient(135deg,#7c6aff,#a855f7); border-radius:50%;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; padding: 4px !important; gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; border-radius: 8px !important;
    color: #64748b !important; font-size: 13px !important;
    font-weight: 500 !important; padding: 8px 16px !important;
    border: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(124,106,255,0.15) !important;
    color: #c4b5fd !important;
}}

/* ── Section title ── */
.section-title {{
    font-size:13px; font-weight:600; color:#94a3b8;
    letter-spacing:0.08em; text-transform:uppercase;
    margin-bottom:14px; display:flex; align-items:center; gap:8px;
}}
.section-title::after {{
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg, rgba(148,163,184,0.2), transparent);
}}

/* ── Prompt buttons ── */
.stButton > button {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(124,106,255,0.2) !important;
    border-radius: 10px !important; color: #94a3b8 !important;
    font-size: 13px !important; font-weight: 400 !important;
    padding: 10px 14px !important; text-align: left !important;
    transition: all 0.2s ease !important;
    white-space: normal !important; height: auto !important; line-height: 1.4 !important;
}}
.stButton > button:hover {{
    background: rgba(124,106,255,0.1) !important;
    border-color: rgba(124,106,255,0.5) !important; color: #c4b5fd !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.15) !important;
}}

/* ── Export buttons ── */
.export-row {{ display:flex; gap:10px; margin:12px 0 20px; flex-wrap:wrap; }}
[data-testid="stDownloadButton"] button {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important; color: #94a3b8 !important;
    font-size: 12px !important; padding: 8px 14px !important;
    transition: all 0.2s !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: rgba(124,106,255,0.1) !important;
    border-color: rgba(124,106,255,0.4) !important; color: #c4b5fd !important;
}}

/* ── Agent log panel ── */
.log-panel {{
    background:rgba(13,17,32,0.8); border:1px solid rgba(124,106,255,0.15);
    border-radius:14px; padding:16px; margin-bottom:20px;
}}
.log-header {{
    font-size:11px; font-weight:600; color:#475569;
    letter-spacing:0.1em; text-transform:uppercase;
    margin-bottom:10px; display:flex; align-items:center; gap:6px;
}}
.log-entry {{
    padding:8px 10px; border-radius:8px; margin:4px 0;
    font-family:'SF Mono','Fira Code',monospace; font-size:12.5px;
    line-height:1.5; color:#94a3b8; border-left:2px solid transparent;
}}
.log-entry.success {{ border-left-color:#4ade80; color:#86efac; background:rgba(74,222,128,0.04); }}
.log-entry.error   {{ border-left-color:#f87171; color:#fca5a5; background:rgba(248,113,113,0.04); }}
.log-entry.retry   {{ border-left-color:#fbbf24; color:#fde68a; background:rgba(251,191,36,0.04); }}

/* ── Metric cards ── */
.metric-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:20px 0; }}
.metric-card {{
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:18px 20px; position:relative; overflow:hidden;
}}
.metric-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#7c6aff,#a855f7,#06b6d4);
}}
.metric-label {{
    font-size:11px; font-weight:600; color:#475569;
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;
}}
.metric-value {{ font-size:22px; font-weight:700; color:#f1f5f9; }}
.metric-value.success-val {{ color:#4ade80; }}
.metric-value.fail-val    {{ color:#f87171; }}

/* ── Schema pill ── */
.schema-pill {{
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:6px; padding:4px 10px; font-size:12px; color:#64748b; margin:2px;
}}
.schema-pill.num {{ border-color:rgba(124,106,255,0.25); color:#a78bfa; }}
.schema-pill.cat {{ border-color:rgba(6,182,212,0.25);   color:#67e8f9; }}

/* ── Dataset card ── */
.dataset-card {{
    background:rgba(124,106,255,0.06); border:1px solid rgba(124,106,255,0.2);
    border-radius:12px; padding:14px 16px; margin-top:12px;
}}
.dataset-card-title {{ font-size:13px; font-weight:600; color:#c4b5fd; margin-bottom:4px; }}
.dataset-card-meta  {{ font-size:12px; color:#64748b; }}

/* ── Conversation history ── */
.conv-turn {{
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:14px 18px; margin-bottom:10px;
}}
.conv-q {{ font-size:13px; font-weight:600; color:#a78bfa; margin-bottom:6px; }}
.conv-s {{ font-size:13px; color:#64748b; line-height:1.6; }}

/* ── Profiler card ── */
.profiler-card {{
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:16px 18px; margin-bottom:10px;
}}
.profiler-col-name {{ font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:4px; }}
.profiler-dtype    {{ font-size:11px; color:#475569; margin-bottom:10px; }}
.profiler-stat     {{ font-size:12px; color:#64748b; margin:2px 0; }}
.profiler-stat span{{ color:#94a3b8; }}
.null-bar-bg {{
    height:4px; background:rgba(255,255,255,0.06);
    border-radius:999px; margin:6px 0;
}}
.null-bar-fill {{
    height:4px; border-radius:999px;
    background:linear-gradient(90deg,#4ade80,#f87171);
}}
.val-tag {{
    display:inline-block; background:rgba(124,106,255,0.1);
    border:1px solid rgba(124,106,255,0.2); border-radius:5px;
    padding:2px 8px; font-size:11px; color:#a78bfa; margin:2px;
}}

/* ── Chat input ── */
[data-testid="stChatInput"] {{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(124,106,255,0.25) !important; border-radius:14px !important;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color:rgba(124,106,255,0.6) !important;
    box-shadow:0 0 0 3px rgba(124,106,255,0.1) !important;
}}

/* ── Query header ── */
.query-header {{
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:16px 20px; margin:20px 0 16px;
    display:flex; align-items:center; gap:12px;
}}
.query-icon {{
    width:34px; height:34px;
    background:linear-gradient(135deg,#7c6aff22,#a855f722);
    border:1px solid rgba(124,106,255,0.3); border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0;
}}
.query-text {{ font-size:16px; font-weight:500; color:#e2e8f0; font-style:italic; }}

/* ── Misc ── */
hr {{ border-color:rgba(255,255,255,0.06) !important; }}
.stCodeBlock {{ border-radius:12px !important; border:1px solid rgba(124,106,255,0.15) !important; }}
.streamlit-expanderHeader {{
    background:rgba(255,255,255,0.03) !important; border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:10px !important; color:#94a3b8 !important; font-size:13px !important;
}}
.stAlert {{ border-radius:10px !important; border:none !important; }}
[data-testid="stDataFrame"] {{
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:12px !important; overflow:hidden !important;
}}
.app-footer {{ text-align:center; padding:32px 0 8px; font-size:12px; color:#1e293b; }}

/* ── Theme overrides ── */
{THEME_CSS}
</style>
""", unsafe_allow_html=True)

# ── Azure credentials ─────────────────────────────────────────────────────────
ENV_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY", "")
ENV_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
ENV_EMBEDDING = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
ENV_LLM       = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-5.1")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">✦</div>
        <div class="sidebar-brand-text">
            <div class="sidebar-brand-title">AI Analyst Agent</div>
            <div class="sidebar-brand-sub">Ciklum AI Academy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Toggle ──────────────────────────────────────────────────────────
    _cur_theme = st.session_state.theme
    _theme_label = "☀️ Light Mode" if _cur_theme == "dark" else "🌙 Dark Mode"
    if st.button(_theme_label, key="theme_toggle"):
        st.session_state.theme = "light" if _cur_theme == "dark" else "dark"
        st.rerun()

    # ── Azure credentials ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🔑 Azure Credentials</div>', unsafe_allow_html=True)

    api_key = st.text_input("API Key", value=ENV_API_KEY, type="password",
                             placeholder="Your Azure OpenAI key…")
    azure_endpoint = st.text_input("Endpoint", value=ENV_ENDPOINT,
                                   placeholder="https://your-resource.cognitiveservices.azure.com")
    col_a, col_b = st.columns(2)
    with col_a:
        embedding_deployment = st.text_input("Embedding Model", value=ENV_EMBEDDING)
    with col_b:
        llm_deployment = st.text_input("LLM Deployment", value=ENV_LLM)

    if ENV_API_KEY:
        st.markdown('<div class="cred-badge">✅ &nbsp;Loaded from .env</div>', unsafe_allow_html=True)

    # ── Cost Tracker display ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">💰 API Cost</div>', unsafe_allow_html=True)
    _ct = st.session_state.cost_tracker
    cost_col, reset_col = st.columns([3, 1])
    with cost_col:
        st.markdown(
            f'<div style="font-size:13px;color:#86efac;padding:6px 0;">'
            f'💰 {_ct.summary_str()}</div>',
            unsafe_allow_html=True,
        )
    with reset_col:
        if st.button("Reset", key="reset_cost"):
            st.session_state.cost_tracker.reset()
            st.rerun()

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📦 Data Source</div>', unsafe_allow_html=True)

    source_type = st.radio(
        "Source",
        ["Built-in Dataset", "Sklearn Dataset", "Upload File", "SQL Connection", "Microsoft Fabric"],
        label_visibility="collapsed",
    )

    df_to_load, desc_to_load, dataset_label = None, None, None

    # ── Built-in ──────────────────────────────────────────────────────────────
    if source_type == "Built-in Dataset":
        _prefill_ds = st.session_state.get("prefill_dataset", "Diamonds")
        _ds_options = ["Diamonds", "Iris", "Tips (Restaurant)", "Gapminder"]
        _ds_index   = _ds_options.index(_prefill_ds) if _prefill_ds in _ds_options else 0
        dataset_choice = st.selectbox("Dataset", _ds_options,
                                      index=_ds_index, label_visibility="collapsed")
        if st.button("⚡  Load Dataset", use_container_width=True, key="load_builtin"):
            if not api_key or not azure_endpoint:
                st.error("Azure credentials required.")
            else:
                with st.spinner("Loading…"):
                    try:
                        df_to_load, desc_to_load = load_builtin(dataset_choice)
                        dataset_label = dataset_choice
                    except Exception as e:
                        st.error(str(e))

    # ── Sklearn ───────────────────────────────────────────────────────────────
    elif source_type == "Sklearn Dataset":
        sklearn_choice = st.selectbox("Dataset",
            ["Diabetes", "Wine", "Breast Cancer", "California Housing"],
            label_visibility="collapsed")
        if st.button("⚡  Load Dataset", use_container_width=True, key="load_sklearn"):
            if not api_key or not azure_endpoint:
                st.error("Azure credentials required.")
            else:
                with st.spinner("Loading sklearn dataset…"):
                    try:
                        df_to_load, desc_to_load = load_sklearn(sklearn_choice)
                        dataset_label = f"sklearn · {sklearn_choice}"
                    except Exception as e:
                        st.error(str(e))

    # ── Upload File ───────────────────────────────────────────────────────────
    elif source_type == "Upload File":
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"],
                                    label_visibility="collapsed")
        if uploaded and st.button("⚡  Load File", use_container_width=True, key="load_file"):
            if not api_key or not azure_endpoint:
                st.error("Azure credentials required.")
            else:
                with st.spinner(f"Reading {uploaded.name}…"):
                    try:
                        df_to_load, desc_to_load = load_file(uploaded, uploaded.name)
                        dataset_label = uploaded.name
                    except Exception as e:
                        st.error(str(e))

    # ── SQL Connection ────────────────────────────────────────────────────────
    elif source_type == "SQL Connection":
        st.caption("SQLAlchemy connection URL")
        sql_conn_str = st.text_input("Connection String",
            placeholder="postgresql://user:pass@host:5432/dbname",
            label_visibility="collapsed")

        nl_sql = st.checkbox("✨ Let AI write the SQL for me")
        if nl_sql:
            sql_nl_input = st.text_area("Describe what data you need",
                placeholder="e.g. Show total sales grouped by region for 2024",
                height=70, label_visibility="collapsed")
            if st.button("Generate SQL", key="gen_sql"):
                if sql_nl_input and sql_conn_str:
                    from tools.code_generator import CodeGeneratorTool
                    try:
                        gen = CodeGeneratorTool(api_key, azure_endpoint, llm_deployment)
                        st.session_state.sql_generated = gen.generate_sql(
                            sql_nl_input, f"Database: {sql_conn_str.split('@')[-1]}",
                            cost_tracker=st.session_state.cost_tracker,
                        )
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Enter connection string and description first.")

        sql_query = st.text_area("SQL Query or Table Name",
            value=st.session_state.sql_generated,
            placeholder="SELECT * FROM my_table  — or just: my_table",
            height=90, label_visibility="collapsed")

        if st.button("⚡  Connect & Load", use_container_width=True, key="load_sql"):
            if not api_key or not azure_endpoint:
                st.error("Azure credentials required.")
            elif not sql_conn_str or not sql_query:
                st.error("Connection string and query are required.")
            else:
                with st.spinner("Connecting…"):
                    try:
                        df_to_load, desc_to_load = load_from_sql(sql_conn_str, sql_query)
                        dataset_label = "SQL Query Result"
                    except Exception as e:
                        st.error(str(e))

    # ── Microsoft Fabric ──────────────────────────────────────────────────────
    elif source_type == "Microsoft Fabric":
        st.caption("Fabric SQL Analytics Endpoint")
        fab_server = st.text_input("Workspace Endpoint",
            placeholder="<workspace-id>.sql.fabric.microsoft.com",
            label_visibility="collapsed")
        fab_db = st.text_input("Lakehouse / Warehouse",
            placeholder="my_lakehouse", label_visibility="collapsed")
        fc1, fc2 = st.columns(2)
        with fc1:
            fab_user = st.text_input("Username", placeholder="user@org.com")
        with fc2:
            fab_pass = st.text_input("Password", type="password", placeholder="••••••••")

        nl_fab = st.checkbox("✨ Let AI write the SQL for me", key="nl_fab")
        if nl_fab:
            fab_nl_input = st.text_area("Describe what data you need",
                placeholder="e.g. Total revenue by product category last 6 months",
                height=70, label_visibility="collapsed", key="fab_nl_area")
            if st.button("Generate SQL", key="gen_fab_sql"):
                if fab_nl_input and fab_db:
                    from tools.code_generator import CodeGeneratorTool
                    try:
                        gen = CodeGeneratorTool(api_key, azure_endpoint, llm_deployment)
                        st.session_state.sql_generated = gen.generate_sql(
                            fab_nl_input, f"Fabric Lakehouse: {fab_db}",
                            cost_tracker=st.session_state.cost_tracker,
                        )
                    except Exception as e:
                        st.error(str(e))

        fab_query = st.text_area("SQL Query or Table Name",
            value=st.session_state.get("sql_generated", ""),
            placeholder="SELECT * FROM dbo.sales  — or just: dbo.sales",
            height=80, label_visibility="collapsed", key="fab_query")

        if st.button("⚡  Connect & Load", use_container_width=True, key="load_fabric"):
            if not api_key or not azure_endpoint:
                st.error("Azure credentials required.")
            elif not all([fab_server, fab_db, fab_user, fab_pass, fab_query]):
                st.error("All Fabric fields are required.")
            else:
                with st.spinner("Connecting to Microsoft Fabric…"):
                    try:
                        df_to_load, desc_to_load = load_from_fabric(
                            fab_server, fab_db, fab_user, fab_pass, fab_query
                        )
                        dataset_label = f"Fabric · {fab_db}"
                    except Exception as e:
                        st.error(str(e))

    # ── Spin up agent + suggest queries ──────────────────────────────────────
    if df_to_load is not None:
        with st.spinner("Building RAG index…"):
            try:
                agent = AnalystAgent(
                    api_key=api_key, df=df_to_load,
                    dataset_name=dataset_label,
                    dataset_description=desc_to_load,
                    azure_endpoint=azure_endpoint,
                    embedding_deployment=embedding_deployment,
                    llm_deployment=llm_deployment,
                    cost_tracker=st.session_state.cost_tracker,
                )
                st.session_state.df           = df_to_load
                st.session_state.dataset_name = dataset_label
                st.session_state.agent        = agent
                st.session_state.onboarding_done = True

                cache_msg = "⚡ Embeddings loaded from cache" if agent.rag_from_cache else "🔢 Embeddings built"
                st.markdown(f'<div class="cache-badge">{cache_msg}</div>', unsafe_allow_html=True)

                with st.spinner("Generating AI query suggestions…"):
                    st.session_state.suggested_queries = agent.suggest_queries()

                st.success(f"✅ {len(df_to_load):,} rows · {len(df_to_load.columns)} cols")
            except Exception as e:
                st.error(str(e))

    # ── Dataset info + schema ─────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        <div class="dataset-card">
            <div class="dataset-card-title">{st.session_state.dataset_name}</div>
            <div class="dataset-card-meta">{len(df):,} rows &nbsp;·&nbsp; {len(df.columns)} columns</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">📋 Schema</div>', unsafe_allow_html=True)
        for col in df.columns:
            dtype  = str(df[col].dtype)
            is_num = "int" in dtype or "float" in dtype
            css    = "num" if is_num else "cat"
            icon   = "⬡" if is_num else "⬠"
            st.markdown(
                f'<span class="schema-pill {css}">{icon} {col} '
                f'<span style="opacity:.5">{dtype}</span></span>',
                unsafe_allow_html=True,
            )

        if st.session_state.agent:
            if st.button("🗑 Clear Conversation Memory", use_container_width=True, key="clear_mem"):
                st.session_state.agent.clear_history()
                st.success("Memory cleared.")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#1e293b;text-align:center;padding-top:8px;">'
        'Azure OpenAI · Plotly · Streamlit</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ &nbsp;AI-Powered Analytics</div>
    <div class="hero-title">Turn Questions into Insights</div>
    <p class="hero-subtitle">
        Ask anything about your data — the agent writes, self-heals, and explains
        the results in plain English.
    </p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-dot"></div>Azure OpenAI · GPT</div>
        <div class="hero-stat"><div class="hero-stat-dot"></div>CSV · Excel · SQL · Fabric</div>
        <div class="hero-stat"><div class="hero-stat-dot"></div>Self-Healing · Memory</div>
        <div class="hero-stat"><div class="hero-stat-dot"></div>Export Charts & Code</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Onboarding Flow ───────────────────────────────────────────────────────────
if not st.session_state.get("onboarding_done") and st.session_state.agent is None:
    st.info(
        "**Welcome! Get started in 3 steps:**\n\n"
        "1. 🔑 Enter your Azure OpenAI credentials in the sidebar\n"
        "2. 📦 Choose a data source and click **Load**\n"
        "3. 💬 Ask a question about your data"
    )
    if st.button("Got it! →", key="onboarding_dismiss"):
        st.session_state.onboarding_done = True
        st.rerun()

# ── Gate ─────────────────────────────────────────────────────────────────────
if st.session_state.agent is None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;background:rgba(255,255,255,0.02);
                border:1px dashed rgba(124,106,255,0.2);border-radius:16px;">
        <div style="font-size:40px;margin-bottom:12px;">📦</div>
        <div style="font-size:16px;font-weight:600;color:#475569;margin-bottom:6px;">No dataset loaded yet</div>
        <div style="font-size:13px;color:#334155;">
            Open the sidebar → enter Azure credentials → choose a data source → click <b>Load</b>.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_analyst, tab_preview, tab_profile, tab_history, tab_dashboard = st.tabs([
    "💬 Analyst", "📋 Data Preview", "📊 Column Profile", "📜 History", "📌 Dashboard"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analyst:

    # ── AI-generated query suggestions ───────────────────────────────────────
    suggestions = st.session_state.suggested_queries
    if suggestions:
        st.markdown('<div class="section-title">✦ AI-Suggested Questions</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.prefill = q

    # ── Conversation memory display ───────────────────────────────────────────
    agent_history = st.session_state.agent.conversation_history
    if agent_history:
        with st.expander(f"🧠 Conversation Memory  ({len(agent_history)} turn{'s' if len(agent_history) > 1 else ''})", expanded=False):
            for turn in agent_history:
                st.markdown(f"""
                <div class="conv-turn">
                    <div class="conv-q">Q: {turn['query']}</div>
                    <div class="conv-s">{turn['summary']}</div>
                </div>""", unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)

    # Pre-fill from query params if available
    _prefill_query = st.session_state.pop("prefill_query", None)
    query = st.chat_input("e.g. Show survival rate by passenger class as a bar chart…")
    if "prefill" in st.session_state:
        query = st.session_state.pop("prefill")
    if _prefill_query and not query:
        query = _prefill_query

    # ── Agent run ─────────────────────────────────────────────────────────────
    if query:
        st.markdown(f"""
        <div class="query-header">
            <div class="query-icon">🔍</div>
            <div class="query-text">{query}</div>
        </div>""", unsafe_allow_html=True)

        # ── Shareable link note ───────────────────────────────────────────────
        _ds_name = st.session_state.dataset_name or ""
        _builtin_names = {"Diamonds", "Iris", "Tips (Restaurant)", "Gapminder"}
        try:
            st.query_params["q"] = query
            if _ds_name in _builtin_names:
                st.query_params["dataset"] = _ds_name
        except Exception:
            pass
        st.markdown(
            '<div style="font-size:11px;color:#475569;margin-bottom:8px;">'
            '🔗 Shareable link updated — copy the browser URL to share this query.</div>',
            unsafe_allow_html=True,
        )

        log_placeholder    = st.empty()
        stream_placeholder = st.empty()
        chart_placeholder  = st.empty()
        logs = []
        _stream_tokens: list[str] = []

        def update_logs():
            entries = "\n".join(
                f'<div class="log-entry {l["type"]}">{l["msg"]}</div>' for l in logs
            )
            log_placeholder.markdown(
                f'<div class="log-panel"><div class="log-header">◎ &nbsp;Agent Log</div>'
                f'{entries}</div>',
                unsafe_allow_html=True,
            )

        def handle_stream(partial_code: str):
            """Callback that receives accumulated code tokens and updates the UI."""
            stream_placeholder.code(partial_code, language="python")

        logs.append({"type": "", "msg": "🔎 Retrieving schema context from RAG index…"})
        update_logs()

        result = st.session_state.agent.run(
            query,
            log_callback=lambda t, m: (logs.append({"type": t, "msg": m}), update_logs()),
            stream_callback=handle_stream,
        )

        # Clear streaming placeholder once full code is available
        stream_placeholder.empty()

        evaluator   = Evaluator()
        eval_result = evaluator.evaluate(
            query=query, code=result.get("code", ""),
            success=result.get("success", False),
            attempts=result.get("attempts", 1), fig=result.get("fig"),
        )

        if result.get("success") and result.get("fig"):
            logs.append({"type": "success", "msg": "✅ Chart rendered successfully!"})
            update_logs()

            fig     = result["fig"]
            summary = result.get("summary", "")
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # ── Pin Chart button ──────────────────────────────────────────────
            if st.button("📌 Pin Chart", key=f"pin_{datetime.now().timestamp()}"):
                st.session_state.pinned_charts.append({
                    "fig":       fig,
                    "query":     query,
                    "summary":   summary,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                st.success("Chart pinned to Dashboard!")

            # ── AI Insight summary ────────────────────────────────────────────
            if summary:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(124,106,255,0.07),rgba(6,182,212,0.05));
                            border:1px solid rgba(124,106,255,0.2);border-radius:14px;
                            padding:20px 24px;margin:4px 0 16px;">
                    <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;
                                text-transform:uppercase;color:#7c6aff;margin-bottom:10px;">
                        ✦ &nbsp;AI Insight
                    </div>
                    <div style="font-size:14.5px;line-height:1.75;color:#cbd5e1;">{summary}</div>
                </div>""", unsafe_allow_html=True)

            # ── Export row ────────────────────────────────────────────────────
            st.markdown('<div class="section-title">⬇ Export</div>', unsafe_allow_html=True)
            ex1, ex2, ex3, ex4, ex5 = st.columns(5)

            with ex1:
                html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode()
                st.download_button(
                    "📊 Chart (.html)", html_bytes,
                    file_name="chart.html", mime="text/html",
                    use_container_width=True,
                )
            with ex2:
                code_bytes = result["code"].encode()
                st.download_button(
                    "🐍 Code (.py)", code_bytes,
                    file_name="analysis.py", mime="text/plain",
                    use_container_width=True,
                )
            with ex3:
                csv_bytes = st.session_state.df.to_csv(index=False).encode()
                st.download_button(
                    "📥 Data (.csv)", csv_bytes,
                    file_name="data.csv", mime="text/csv",
                    use_container_width=True,
                )
            with ex4:
                # PNG export via kaleido
                try:
                    png_bytes = fig.to_image(format="png")
                    st.download_button(
                        "🖼 Chart (.png)", png_bytes,
                        file_name="chart.png", mime="image/png",
                        use_container_width=True,
                    )
                except ImportError:
                    st.warning("Install kaleido for PNG: pip install kaleido")
                except Exception as e:
                    st.warning(f"PNG export failed: {e}")

            with ex5:
                # PDF export via fpdf2
                try:
                    from fpdf import FPDF
                    import io

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, "AI Analyst Agent — Report", ln=True)
                    pdf.ln(4)

                    pdf.set_font("Helvetica", "B", 11)
                    pdf.cell(0, 8, f"Dataset: {st.session_state.dataset_name or '—'}", ln=True)
                    pdf.ln(2)

                    pdf.set_font("Helvetica", "B", 11)
                    pdf.cell(0, 8, "Query:", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, query)
                    pdf.ln(4)

                    if summary:
                        pdf.set_font("Helvetica", "B", 11)
                        pdf.cell(0, 8, "AI Insight:", ln=True)
                        pdf.set_font("Helvetica", "", 10)
                        pdf.multi_cell(0, 6, summary)
                        pdf.ln(4)

                    pdf.set_font("Helvetica", "", 9)
                    attempts_val  = result.get("attempts", 1)
                    score_val     = eval_result.get("score", "—")
                    pdf.cell(0, 6, f"Attempts: {attempts_val}/3   Score: {score_val}/10   "
                                   f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                    pdf.ln(4)

                    # Try to include PNG chart image if kaleido available
                    try:
                        png_bytes_for_pdf = fig.to_image(format="png")
                        tmp_path = "/tmp/_ai_analyst_chart.png"
                        with open(tmp_path, "wb") as f_img:
                            f_img.write(png_bytes_for_pdf)
                        pdf.image(tmp_path, x=10, w=190)
                    except Exception:
                        pdf.set_font("Helvetica", "I", 9)
                        pdf.cell(0, 6, "(Install kaleido to include chart image in PDF)", ln=True)

                    pdf_bytes = bytes(pdf.output())
                    st.download_button(
                        "📄 Report (.pdf)", pdf_bytes,
                        file_name="report.pdf", mime="application/pdf",
                        use_container_width=True,
                    )
                except ImportError:
                    st.warning("Install fpdf2 for PDF export: pip install fpdf2")
                except Exception as e:
                    st.warning(f"PDF export failed: {e}")

        else:
            logs.append({"type": "error", "msg": f"❌ Failed after {result.get('attempts', 3)} attempts."})
            update_logs()
            if result.get("last_error"):
                st.error(f"**Last error:** {result['last_error']}")

        # ── Metrics ───────────────────────────────────────────────────────────
        success     = result["success"]
        attempts    = result.get("attempts", 1)
        score       = eval_result["score"]
        self_healed = attempts > 1 and success

        status_css  = "success-val" if success     else "fail-val"
        score_css   = "success-val" if score >= 7  else ("fail-val" if score < 5 else "")
        heal_css    = "success-val" if self_healed else ""

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value {status_css}">{"✅ Success" if success else "❌ Failed"}</div>
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
                <div class="metric-value {heal_css}">{"✦ Yes" if self_healed else "— No"}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if result.get("code"):
                with st.expander("🧾 Generated Python Code"):
                    st.code(result["code"], language="python")
        with c2:
            with st.expander("📈 Evaluation Breakdown"):
                st.json(eval_result)

        # ── Save to persistent history ────────────────────────────────────────
        entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Dataset":   st.session_state.dataset_name,
            "Query":     query,
            "Status":    "✅" if success else "❌",
            "Attempts":  attempts,
            "Score":     f"{score}/10",
            "Self-healed": "Yes" if self_healed else "No",
            "Chart Type":  eval_result.get("chart_type_detected", "—"),
        }
        st.session_state.history.append(entry)
        _append_history(entry)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_preview:
    df = st.session_state.df
    st.markdown('<div class="section-title">📋 Data Preview</div>', unsafe_allow_html=True)

    rows_per_page = st.select_slider(
        "Rows per page", options=[25, 50, 100, 250, 500], value=50,
    )
    total_pages = max(1, (len(df) - 1) // rows_per_page + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * rows_per_page
    end   = start + rows_per_page

    st.caption(f"Showing rows {start + 1}–{min(end, len(df))} of {len(df):,}")
    st.dataframe(df.iloc[start:end], use_container_width=True, hide_index=False)

    dl_csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download full dataset (.csv)", dl_csv,
                       file_name="dataset.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COLUMN PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    import plotly.express as _px

    df = st.session_state.df
    st.markdown('<div class="section-title">📊 Column Profile</div>', unsafe_allow_html=True)

    n_numeric = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    n_categ   = len(df.columns) - n_numeric
    n_nulls   = df.isnull().any(axis=1).sum()

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Rows",    f"{len(df):,}")
    p2.metric("Total Columns", len(df.columns))
    p3.metric("Numeric Cols",  n_numeric)
    p4.metric("Rows with Nulls", f"{n_nulls:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    for idx, col in enumerate(df.columns):
        dtype     = str(df[col].dtype)
        null_pct  = df[col].isna().mean() * 100
        null_fill = null_pct
        container = col_left if idx % 2 == 0 else col_right

        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].describe()
            container.markdown(f"""
            <div class="profiler-card">
                <div class="profiler-col-name">⬡ {col}</div>
                <div class="profiler-dtype">numeric · {dtype}</div>
                <div class="profiler-stat">Min <span>{s['min']:.3g}</span> &nbsp;·&nbsp;
                    Max <span>{s['max']:.3g}</span> &nbsp;·&nbsp;
                    Mean <span>{s['mean']:.3g}</span> &nbsp;·&nbsp;
                    Std <span>{s['std']:.3g}</span></div>
                <div class="profiler-stat">Unique <span>{df[col].nunique():,}</span>
                    &nbsp;·&nbsp; Nulls <span>{null_pct:.1f}%</span></div>
                <div class="null-bar-bg">
                    <div class="null-bar-fill" style="width:{null_fill:.1f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)
            # Distribution sparkline
            try:
                mini_fig = _px.histogram(
                    df[col].dropna(), x=col,
                    template="plotly_dark",
                    height=120,
                    labels={col: ""},
                )
                mini_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(showticklabels=False, title=""),
                    yaxis=dict(showticklabels=False, title=""),
                )
                container.plotly_chart(
                    mini_fig, use_container_width=True,
                    config={"displayModeBar": False},
                )
            except Exception:
                pass
        else:
            top = df[col].value_counts().head(5)
            tags = "".join(f'<span class="val-tag">{v}</span>' for v in top.index)
            container.markdown(f"""
            <div class="profiler-card">
                <div class="profiler-col-name">⬠ {col}</div>
                <div class="profiler-dtype">categorical · {dtype}</div>
                <div class="profiler-stat">Unique <span>{df[col].nunique():,}</span>
                    &nbsp;·&nbsp; Nulls <span>{null_pct:.1f}%</span></div>
                <div class="null-bar-bg">
                    <div class="null-bar-fill" style="width:{null_fill:.1f}%"></div>
                </div>
                <div style="margin-top:8px">{tags}</div>
            </div>""", unsafe_allow_html=True)
            # Top-8 bar sparkline
            try:
                top8 = df[col].value_counts().head(8).reset_index()
                top8.columns = ["value", "count"]
                mini_fig = _px.bar(
                    top8, x="value", y="count",
                    template="plotly_dark",
                    height=120,
                    labels={"value": "", "count": ""},
                )
                mini_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(showticklabels=False, title=""),
                    yaxis=dict(showticklabels=False, title=""),
                )
                container.plotly_chart(
                    mini_fig, use_container_width=True,
                    config={"displayModeBar": False},
                )
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-title">📜 Query History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        hist_bytes = hist_df.to_csv(index=False).encode()
        st.download_button("📥 Export history (.csv)", hist_bytes,
                           file_name="query_history.csv", mime="text/csv")

        if st.button("🗑 Clear session history", key="clear_hist"):
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared.")
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#334155;">
            No queries yet — run some analyses and they'll appear here.
        </div>""", unsafe_allow_html=True)

    # ── Error Log ─────────────────────────────────────────────────────────────
    recent_errors = _load_errors(20)
    with st.expander(f"🔴 Error Log ({len(recent_errors)} recent errors)"):
        if recent_errors:
            for err in reversed(recent_errors):
                st.markdown(f"""
                <div style="background:rgba(248,113,113,0.05);border:1px solid rgba(248,113,113,0.15);
                            border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:12px;">
                    <div style="color:#fca5a5;font-weight:600;">{err.get('timestamp','—')} · {err.get('dataset','—')}</div>
                    <div style="color:#f87171;margin:4px 0;">{err.get('error_message','—')}</div>
                    <div style="color:#64748b;font-style:italic;">Query: {err.get('query','—')}</div>
                    <div style="color:#475569;font-family:monospace;margin-top:6px;white-space:pre-wrap;">{err.get('code_snippet','')[:200]}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;padding:8px;">No errors logged.</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DASHBOARD (Pinned Charts)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    import plotly.io as pio

    pinned = st.session_state.pinned_charts
    st.markdown('<div class="section-title">📌 Pinned Charts Dashboard</div>', unsafe_allow_html=True)

    if not pinned:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:rgba(255,255,255,0.02);
                    border:1px dashed rgba(124,106,255,0.2);border-radius:16px;">
            <div style="font-size:36px;margin-bottom:12px;">📌</div>
            <div style="font-size:15px;font-weight:600;color:#475569;margin-bottom:6px;">No charts pinned yet</div>
            <div style="font-size:13px;color:#334155;">
                Run a query in the Analyst tab and click <b>📌 Pin Chart</b> to add it here.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Download Dashboard button ──────────────────────────────────────────
        try:
            _combined_html_parts = ["<html><head><meta charset='utf-8'>"
                                    "<title>AI Analyst Dashboard</title></head><body>"
                                    "<h1 style='font-family:sans-serif;'>AI Analyst Dashboard</h1>"]
            for pc in pinned:
                _combined_html_parts.append(
                    f"<h3 style='font-family:sans-serif;'>{pc['query']}</h3>"
                    + pc["fig"].to_html(full_html=False, include_plotlyjs="cdn")
                    + f"<p style='font-family:sans-serif;color:#555;'>{pc.get('summary','')}</p><hr>"
                )
            _combined_html_parts.append("</body></html>")
            dashboard_html = "\n".join(_combined_html_parts).encode()
            st.download_button(
                "⬇ Download Dashboard (.html)", dashboard_html,
                file_name="dashboard.html", mime="text/html",
            )
        except Exception as e:
            st.warning(f"Dashboard export failed: {e}")

        # ── Render pinned charts in 2-column grid ─────────────────────────────
        for i in range(0, len(pinned), 2):
            row_cols = st.columns(2)
            for j, col_slot in enumerate(row_cols):
                chart_idx = i + j
                if chart_idx >= len(pinned):
                    break
                pc = pinned[chart_idx]
                with col_slot:
                    st.markdown(
                        f'<div style="font-size:12px;color:#a78bfa;margin-bottom:4px;">'
                        f'{pc["timestamp"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:8px;">'
                        f'{pc["query"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(pc["fig"], use_container_width=True,
                                    key=f"pinned_{chart_idx}")
                    if pc.get("summary"):
                        st.caption(pc["summary"])
                    if st.button("Unpin", key=f"unpin_{chart_idx}"):
                        st.session_state.pinned_charts.pop(chart_idx)
                        st.rerun()

        # ── Chart Comparison View ──────────────────────────────────────────────
        if len(pinned) >= 2:
            st.markdown("---")
            st.markdown('<div class="section-title">🔀 Compare Charts</div>', unsafe_allow_html=True)
            chart_labels = [f"{i+1}. {pc['query'][:60]}" for i, pc in enumerate(pinned)]

            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                sel_a = st.selectbox("Chart A", options=range(len(pinned)),
                                     format_func=lambda i: chart_labels[i],
                                     key="compare_a")
            with cmp_col2:
                sel_b = st.selectbox("Chart B", options=range(len(pinned)),
                                     format_func=lambda i: chart_labels[i],
                                     index=min(1, len(pinned) - 1),
                                     key="compare_b")

            side_a, side_b = st.columns(2)
            with side_a:
                st.markdown(f'**{pinned[sel_a]["query"]}**')
                st.plotly_chart(pinned[sel_a]["fig"], use_container_width=True,
                                key="cmp_chart_a")
            with side_b:
                st.markdown(f'**{pinned[sel_b]["query"]}**')
                st.plotly_chart(pinned[sel_b]["fig"], use_container_width=True,
                                key="cmp_chart_b")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Azure OpenAI · Plotly · Streamlit &nbsp;·&nbsp; Ciklum AI Academy Capstone
</div>""", unsafe_allow_html=True)
