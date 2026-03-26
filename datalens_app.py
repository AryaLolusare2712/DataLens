import os
import io
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens – AI Analytics",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark theme, custom fonts)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d0f17;
    --surface:   #13151f;
    --surface2:  #1a1d2e;
    --border:    #252840;
    --accent:    #7c6ff7;
    --accent2:   #00e5c3;
    --accent3:   #ff6b9d;
    --text:      #e8eaf6;
    --muted:     #6b7280;
    --live:      #22c55e;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* main area */
.main .block-container { padding: 1.5rem 2rem 3rem; }

/* headings */
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; }

/* metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
}
.metric-card.rev::before { background: linear-gradient(90deg,#7c6ff7,#a78bfa); }
.metric-card.exp::before { background: linear-gradient(90deg,#00e5c3,#06b6d4); }
.metric-card.pro::before { background: linear-gradient(90deg,#ff6b9d,#f97316); }
.metric-card.uni::before { background: linear-gradient(90deg,#facc15,#f97316); }
.metric-label { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin-bottom:8px; }
.metric-value { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; line-height:1; margin-bottom:6px; }
.metric-sub { font-size:11px; color:var(--muted); }
.metric-icon { position:absolute; top:18px; right:18px; font-size:1.6rem; opacity:.6; }

/* section headers */
.section-header {
    font-family:'Syne',sans-serif;
    font-size:1.3rem; font-weight:700;
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    margin: 24px 0 14px;
}

/* profile card */
.profile-card {
    background: var(--surface2);
    border:1px solid var(--border);
    border-radius:16px;
    padding:20px;
}
.profile-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid var(--border); }
.profile-row:last-child { border-bottom:none; }
.profile-key { color:var(--muted); font-size:.85rem; }
.profile-val { font-family:'Space Mono',monospace; font-size:.85rem; }

/* col type badge */
.col-badge {
    background: var(--surface);
    border:1px solid var(--border);
    border-radius:10px;
    padding:12px 14px;
    text-align:center;
    font-size:.78rem;
}
.col-badge .col-name { font-weight:700; font-size:.9rem; margin-bottom:4px; }
.col-badge .col-type { color:var(--muted); font-size:.72rem; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; }
.col-complete { color:#22c55e; font-size:.75rem; }

/* insight card */
.insight-card {
    background: var(--surface2);
    border:1px solid var(--border);
    border-radius:16px;
    padding:20px;
    height:100%;
}
.insight-badge {
    display:inline-block;
    padding:4px 12px;
    border-radius:20px;
    font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:1.5px;
    font-weight:700; text-transform:uppercase;
    margin-bottom:12px;
}
.badge-trend { background:#1e3a2f; color:#22c55e; }
.badge-insight { background:#2d1b69; color:#a78bfa; }
.badge-alert { background:#3a1a00; color:#f97316; }
.badge-correlation { background:#0f2a2a; color:#00e5c3; }
.insight-text { font-size:.88rem; line-height:1.6; color:#c4c9e0; margin-bottom:16px; text-align:center; }
.insight-bar-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-size:.8rem; }
.insight-bar-label { color:var(--muted); width:60px; }
.insight-bar-track { flex:1; height:6px; background:var(--surface); border-radius:3px; margin:0 10px; overflow:hidden; }
.insight-bar-fill-mean { background:var(--accent); height:100%; border-radius:3px; }
.insight-bar-fill-max  { background:var(--accent2); height:100%; border-radius:3px; }
.insight-bar-fill-std  { background:#facc15; height:100%; border-radius:3px; }
.insight-bar-val { color:var(--text); width:50px; text-align:right; font-family:'Space Mono',monospace; font-size:.75rem; }

/* chat */
.chat-bubble-user {
    background: var(--accent);
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; margin: 6px 0 6px auto;
    max-width: 70%; font-size:.88rem;
    text-align:right;
}
.chat-bubble-ai {
    background: var(--surface2);
    border:1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 10px 16px; margin: 6px auto 6px 0;
    max-width: 85%; font-size:.88rem;
}
.chat-avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1rem; }

/* upload zone */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 20px;
    padding: 60px 20px;
    text-align: center;
    background: var(--surface2);
}
.upload-title { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; margin:16px 0 8px; }
.upload-sub { color:var(--muted); font-size:.88rem; margin-bottom:16px; }
.file-badge {
    display:inline-block;
    border:1px solid var(--border);
    border-radius:20px;
    padding:4px 14px;
    font-family:'Space Mono',monospace;
    font-size:.78rem; margin:3px;
    color:var(--muted);
}

/* nav item active */
div[data-testid="stSidebarNav"] { display:none; }

/* stButton */
div.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .78rem !important;
    letter-spacing: 1.5px !important;
    padding: 10px 20px !important;
    transition: opacity .2s !important;
}
div.stButton > button:hover { opacity:.85 !important; }

/* stTextInput */
div[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border:1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important;
}

/* stFileUploader */
div[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border:2px dashed var(--border) !important;
    border-radius:16px !important;
}

/* table */
div[data-testid="stDataFrame"] { background: var(--surface2) !important; border-radius:12px !important; }

/* selectbox */
div[data-testid="stSelectbox"] > div { background:var(--surface2) !important; border-color:var(--border) !important; }

/* multiselect */
div[data-testid="stMultiSelect"] > div { background:var(--surface2) !important; }

/* scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:10px; }

/* top bar */
.top-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding: 0 0 18px; border-bottom:1px solid var(--border); margin-bottom:24px;
}
.top-bar-title { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; }
.top-bar-right { display:flex; gap:10px; align-items:center; }
.badge-gemini {
    background:#2d1b69; color:#a78bfa;
    border:1px solid #4c3d9e;
    border-radius:20px; padding:5px 14px;
    font-family:'Space Mono',monospace; font-size:.72rem; font-weight:700;
}
.badge-live {
    background:#1e3a2f; color:#22c55e;
    border:1px solid #166534;
    border-radius:20px; padding:5px 14px;
    font-family:'Space Mono',monospace; font-size:.72rem; font-weight:700;
}

/* explorer table */
.explorer-table { width:100%; border-collapse:collapse; font-size:.85rem; }
.explorer-table th {
    font-family:'Space Mono',monospace; font-size:.7rem;
    letter-spacing:1px; text-transform:uppercase;
    color:var(--muted); padding:10px 14px;
    border-bottom:1px solid var(--border);
    text-align:left;
}
.explorer-table td { padding:10px 14px; border-bottom:1px solid var(--border); }
.explorer-table tr:hover td { background:var(--surface2); }
.explorer-table .num { color:#a78bfa; font-family:'Space Mono',monospace; }
.explorer-table .text-cell { color:var(--text); }

/* heatmap cell */
.heatmap-grid { display:grid; gap:4px; }

/* suggestion chips */
.chip {
    display:inline-block;
    border:1px solid var(--border);
    border-radius:20px;
    padding:5px 14px;
    font-size:.78rem; margin:3px;
    color:var(--muted); cursor:pointer;
    background:var(--surface2);
    transition: border-color .2s, color .2s;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in [("df", None), ("summary", None), ("chat_history", []),
              ("page", "Upload"), ("filename", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  GEMINI SETUP
# ─────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")

def get_gemini_model():
    key = st.session_state.get("api_key", GEMINI_KEY)
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.0-flash")

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def fmt_num(n):
    if abs(n) >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if abs(n) >= 1_000: return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"

def analyze_df(df):
    numeric = df.select_dtypes(include=[np.number])
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "numeric_cols": numeric.columns.tolist(),
        "text_cols": df.select_dtypes(include=["object","category"]).columns.tolist(),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_summary": numeric.describe().to_dict() if not numeric.empty else {},
    }
    try:
        corr = numeric.corr()
        pairs = []
        for a in corr.columns:
            for b in corr.columns:
                if a < b:
                    pairs.append((a, b, float(corr.loc[a, b])))
        summary["top_correlations"] = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        summary["corr_matrix"] = corr.to_dict()
    except Exception:
        summary["top_correlations"] = []
        summary["corr_matrix"] = {}
    return summary

SAMPLE_DATA = pd.DataFrame({
    "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "Revenue": [37017,40539,58683,65341,57977,67708,48956,48123,46312,56423,86968,86418],
    "Expenses": [20343,24971,21841,23349,32760,33733,30208,31013,28456,29871,35282,34189],
    "Profit": [27037,16442,23519,29839,22653,30760,29642,25506,22108,32415,37415,36192],
    "Units": [642,504,572,486,770,667,614,705,598,732,770,744],
})

def build_prompt(question, summary):
    parts = [
        f"Dataset: {summary['shape'][0]} rows × {summary['shape'][1]} cols",
        f"Columns: {', '.join(summary['columns'][:15])}",
        f"Numeric: {', '.join(summary['numeric_cols'])}",
        f"Missing values: {summary['missing_total']}",
        "Stats:",
    ]
    for col, stats in summary.get("numeric_summary", {}).items():
        mean = stats.get("mean", 0)
        mx = stats.get("max", 0)
        mn = stats.get("min", 0)
        parts.append(f" - {col}: mean={mean:.1f}, min={mn:.1f}, max={mx:.1f}")
    parts.append("Top correlations:")
    for a, b, v in summary.get("top_correlations", []):
        parts.append(f" - {a} ↔ {b}: {v:.2f}")
    parts.append(f"\nUser Question: {question}")
    return "\n".join(parts)

def ask_gemini(question):
    model = get_gemini_model()
    if not model:
        return "⚠️ No API key configured. Enter your Gemini API key in the sidebar."
    summary = st.session_state.summary
    if not summary:
        return "📂 Please upload a dataset first."
    prompt = build_prompt(question, summary)
    sys = "You are a world-class data analyst. Answer clearly and concisely. Explain patterns, trends, and correlations. Be direct and insightful."
    try:
        resp = model.generate_content(f"{sys}\n\n{prompt}")
        return resp.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8eaf6", size=12),
    xaxis=dict(gridcolor="#252840", linecolor="#252840", tickcolor="#252840"),
    yaxis=dict(gridcolor="#252840", linecolor="#252840", tickcolor="#252840"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#252840"),
    margin=dict(l=10, r=10, t=30, b=10),
)
COLORS = ["#7c6ff7","#00e5c3","#ff6b9d","#facc15","#f97316","#06b6d4"]

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 10px">
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                    background:linear-gradient(90deg,#7c6ff7,#00e5c3);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            DataLens
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:.65rem;
                    letter-spacing:3px;color:#6b7280;margin-top:2px;">AI ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#6b7280;font-size:.72rem;letter-spacing:2px;margin:16px 0 8px;font-family:Space Mono,monospace'>WORKSPACE</div>", unsafe_allow_html=True)

    pages = [
        ("⬆", "Upload"), ("◎", "Overview"), ("▦", "Charts"),
        ("⊞", "Heatmap"), ("✦", "AI Insights"), ("▭", "Explorer"), ("◆", "AI Chat")
    ]
    for icon, name in pages:
        active = st.session_state.page == name
        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if active else "secondary"
        ):
            st.session_state.page = name
            st.rerun()

    st.markdown("<div style='color:#6b7280;font-size:.72rem;letter-spacing:2px;margin:16px 0 8px;font-family:Space Mono,monospace'>QUICK ACTIONS</div>", unsafe_allow_html=True)
    if st.button("◇  Load sample data", use_container_width=True):
        st.session_state.df = SAMPLE_DATA.copy()
        st.session_state.summary = analyze_df(SAMPLE_DATA)
        st.session_state.filename = "sample_data.csv"
        st.session_state.page = "Overview"
        st.rerun()

    st.markdown("<hr style='border-color:#252840;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#6b7280;font-size:.72rem;letter-spacing:2px;margin-bottom:8px;font-family:Space Mono,monospace'>GEMINI API KEY</div>", unsafe_allow_html=True)
    api_key_input = st.text_input("", type="password", placeholder="AIza...", label_visibility="collapsed", key="api_key")

# ─────────────────────────────────────────────
#  TOP BAR  (shared)
# ─────────────────────────────────────────────
def top_bar(title):
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-title">{title}</div>
        <div class="top-bar-right">
            <span class="badge-gemini">GEMINI AI</span>
            <span class="badge-live">● LIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  PAGE: UPLOAD
# ═══════════════════════════════════════════
if st.session_state.page == "Upload":
    top_bar("Upload Dataset")
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:3rem">📊</div>
        <div class="upload-title">Drop your dataset here</div>
        <div class="upload-sub">Drag & drop or click to browse<br>CSV and Excel files supported</div>
        <div>
            <span class="file-badge">CSV</span>
            <span class="file-badge">XLSX</span>
            <span class="file-badge">XLS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.session_state.summary = analyze_df(df)
            st.session_state.filename = uploaded.name
            st.success(f"✅ **{uploaded.name}** loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            if st.button("→ Go to Overview"):
                st.session_state.page = "Overview"
                st.rerun()
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

    st.markdown("<div style='text-align:center;color:#6b7280;margin-top:16px;font-size:.85rem'>— or — <a href='#' style='color:#7c6ff7;text-decoration:none'>load sample dataset</a></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Load Sample Dataset", use_container_width=True):
            st.session_state.df = SAMPLE_DATA.copy()
            st.session_state.summary = analyze_df(SAMPLE_DATA)
            st.session_state.filename = "sample_data.csv"
            st.session_state.page = "Overview"
            st.rerun()

# ═══════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════
elif st.session_state.page == "Overview":
    top_bar("Dataset Overview")
    df = st.session_state.df
    if df is None:
        st.info("📂 Upload a dataset first.")
    else:
        s = st.session_state.summary
        num_cols = s["numeric_cols"]

        # KPI cards
        icons = {"Revenue":"💰","Expenses":"📉","Profit":"🎯","Units":"⚡"}
        cls   = {"Revenue":"rev","Expenses":"exp","Profit":"pro","Units":"uni"}
        col_cards = st.columns(min(len(num_cols), 4))
        for i, col in enumerate(num_cols[:4]):
            with col_cards[i]:
                total = df[col].sum()
                avg   = df[col].mean()
                mx    = df[col].max()
                icon  = icons.get(col, "📊")
                c     = cls.get(col, "rev")
                st.markdown(f"""
                <div class="metric-card {c}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{col}</div>
                    <div class="metric-value">{fmt_num(total)}</div>
                    <div class="metric-sub">avg {fmt_num(avg)} · max {fmt_num(mx)}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Dataset Profile</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#6b7280;font-size:.8rem;margin-bottom:12px;font-family:Space Mono,monospace'>{s['filename'] if 'filename' in s else st.session_state.filename}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            rows = [
                ("Total rows", s["shape"][0], "#7c6ff7"),
                ("Total columns", s["shape"][1], "#00e5c3"),
                ("Numeric columns", len(s["numeric_cols"]), "#00e5c3"),
                ("Text columns", len(s["text_cols"]), "#ff6b9d"),
                ("Missing values", s["missing_total"], "#22c55e" if s["missing_total"]==0 else "#f97316"),
            ]
            html = "<div class='profile-card'><div style='font-size:.95rem;font-weight:600;margin-bottom:8px;font-family:Syne,sans-serif'>Overview</div>"
            for label, val, color in rows:
                html += f"<div class='profile-row'><span class='profile-key'>{label}</span><span class='profile-val' style='color:{color}'>{val}</span></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        with c2:
            html = "<div class='profile-card'><div style='font-size:.95rem;font-weight:600;margin-bottom:12px;font-family:Syne,sans-serif'>Column types</div><div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>"
            for col in df.columns:
                dtype = str(df[col].dtype)
                is_num = col in s["numeric_cols"]
                type_label = "NUMERIC" if is_num else "TEXT"
                html += f"<div class='col-badge'><div class='col-name'>{col}</div><div class='col-type'>{type_label}</div><div class='col-complete'>✓ complete</div></div>"
            html += "</div></div>"
            st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  PAGE: CHARTS
# ═══════════════════════════════════════════
elif st.session_state.page == "Charts":
    top_bar("Visualizations")
    df = st.session_state.df
    if df is None:
        st.info("📂 Upload a dataset first.")
    else:
        s = st.session_state.summary
        num_cols = s["numeric_cols"]
        text_cols = s["text_cols"]

        c1, c2 = st.columns([3,1])
        with c1:
            selected_cols = st.multiselect("Columns", num_cols, default=num_cols[:2] if len(num_cols)>=2 else num_cols)
        with c2:
            chart_type = st.selectbox("Type", ["bar","line","area","pie"], label_visibility="visible")

        x_col = text_cols[0] if text_cols else df.index.astype(str)
        x_data = df[text_cols[0]] if text_cols else df.index.astype(str)

        if selected_cols:
            fig = go.Figure()
            for i, col in enumerate(selected_cols):
                color = COLORS[i % len(COLORS)]
                if chart_type == "bar":
                    fig.add_trace(go.Bar(x=x_data, y=df[col], name=col, marker_color=color, marker_line_width=0))
                elif chart_type == "line":
                    fig.add_trace(go.Scatter(x=x_data, y=df[col], name=col, line=dict(color=color, width=2.5), mode="lines+markers"))
                elif chart_type == "area":
                    if color.startswith("#"):
                        r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                        fill_color = f"rgba({r},{g},{b},0.15)"
                    elif "rgb(" in color:
                        fill_color = color.replace("rgb(", "rgba(").replace(")", ",0.15)")
                    else:
                        fill_color = color
                    fig.add_trace(go.Scatter(x=x_data, y=df[col], name=col, fill="tozeroy", line=dict(color=color, width=2), fillcolor=fill_color))
                elif chart_type == "pie":
                    fig.add_trace(go.Pie(labels=x_data, values=df[col], name=col, marker=dict(colors=COLORS)))
                    break

            if chart_type == "bar":
                fig.update_layout(barmode="group", **PLOT_LAYOUT)
            else:
                fig.update_layout(**PLOT_LAYOUT)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # small stat row
        if num_cols:
            st.markdown("<div class='section-header' style='font-size:.95rem'>Quick Stats</div>", unsafe_allow_html=True)
            cols = st.columns(len(num_cols[:4]))
            for i, col in enumerate(num_cols[:4]):
                with cols[i]:
                    st.metric(col, fmt_num(df[col].sum()), f"avg {fmt_num(df[col].mean())}")

# ═══════════════════════════════════════════
#  PAGE: HEATMAP
# ═══════════════════════════════════════════
elif st.session_state.page == "Heatmap":
    top_bar("Correlation Heatmap")
    df = st.session_state.df
    if df is None:
        st.info("📂 Upload a dataset first.")
    else:
        s = st.session_state.summary
        num_cols = s["numeric_cols"]
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation.")
        else:
            corr = df[num_cols].corr()
            st.markdown("<div style='color:#6b7280;font-size:.82rem;margin-bottom:16px'>Pearson correlation between numeric columns</div>", unsafe_allow_html=True)

            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale=[[0,"#e53e3e"],[0.5,"#1a1d2e"],[1,"#7c6ff7"]],
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=14, color="white", family="Space Mono"),
                hoverongaps=False,
                showscale=True,
                colorbar=dict(
                    tickcolor="#252840", tickfont=dict(color="#6b7280"),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                ),
            ))
            heatmap_layout = {**PLOT_LAYOUT, "height": 420}
            heatmap_layout["xaxis"] = {**PLOT_LAYOUT.get("xaxis", {}), "side": "top", "gridcolor": "rgba(0,0,0,0)"}
            heatmap_layout["yaxis"] = {**PLOT_LAYOUT.get("yaxis", {}), "gridcolor": "rgba(0,0,0,0)"}
            fig.update_layout(**heatmap_layout)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # legend
            st.markdown("""
            <div style='text-align:center;font-size:.75rem;color:#6b7280;margin-top:8px;font-family:Space Mono,monospace'>
                <span style='background:linear-gradient(90deg,#e53e3e,#1a1d2e,#7c6ff7);
                    display:inline-block;width:160px;height:8px;border-radius:4px;
                    vertical-align:middle;margin:0 10px'></span>
                Purple = positive · Red = negative correlation
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  PAGE: AI INSIGHTS
# ═══════════════════════════════════════════
elif st.session_state.page == "AI Insights":
    top_bar("AI Insights")
    df = st.session_state.df
    if df is None:
        st.info("📂 Upload a dataset first.")
    else:
        s = st.session_state.summary
        num_cols = s["numeric_cols"]
        st.markdown("<div style='color:#6b7280;font-size:.82rem;margin-bottom:20px'>Auto-generated from dataset statistics</div>", unsafe_allow_html=True)

        badge_types = [
            ("TREND","badge-trend"), ("INSIGHT","badge-insight"),
            ("ALERT","badge-alert"), ("CORRELATION","badge-correlation"),
        ]

        cols = st.columns(2)
        for i, col in enumerate(num_cols[:4]):
            stats = s["numeric_summary"].get(col, {})
            mean_v = stats.get("mean", 0)
            max_v  = stats.get("max", 0)
            min_v  = stats.get("min", 0)
            std_v  = stats.get("std", 0)
            total  = df[col].sum()
            pct_above = int(round((df[col] > mean_v).mean() * 100))
            badge_name, badge_cls = badge_types[i % len(badge_types)]
            safe_max = max_v if max_v != 0 else 1

            with cols[i % 2]:
                bar_mean = min(100, int(mean_v / safe_max * 100))
                bar_std  = min(100, int(std_v  / safe_max * 100))

                st.markdown(f"""
                <div class="insight-card" style="margin-bottom:16px">
                    <div style="text-align:center">
                        <span class="insight-badge {badge_cls}">{badge_name}</span>
                    </div>
                    <div class="insight-text">
                        <b>{col}</b> shows an average of <b style="color:#00e5c3">{fmt_num(mean_v)}</b>
                        with values ranging from {fmt_num(min_v)} to {fmt_num(max_v)}.
                        Standard deviation is {fmt_num(std_v)}, and {pct_above}% of values exceed the mean.
                    </div>
                    <div class="insight-bar-row">
                        <span class="insight-bar-label">Mean</span>
                        <div class="insight-bar-track"><div class="insight-bar-fill-mean" style="width:{bar_mean}%"></div></div>
                        <span class="insight-bar-val">{fmt_num(mean_v)}</span>
                    </div>
                    <div class="insight-bar-row">
                        <span class="insight-bar-label">Max</span>
                        <div class="insight-bar-track"><div class="insight-bar-fill-max" style="width:100%"></div></div>
                        <span class="insight-bar-val">{fmt_num(max_v)}</span>
                    </div>
                    <div class="insight-bar-row">
                        <span class="insight-bar-label">Std dev</span>
                        <div class="insight-bar-track"><div class="insight-bar-fill-std" style="width:{bar_std}%"></div></div>
                        <span class="insight-bar-val">{fmt_num(std_v)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  PAGE: EXPLORER
# ═══════════════════════════════════════════
elif st.session_state.page == "Explorer":
    top_bar("Data Explorer")
    df = st.session_state.df
    if df is None:
        st.info("📂 Upload a dataset first.")
    else:
        st.markdown(f"<div style='color:#6b7280;font-size:.8rem;margin-bottom:16px;font-family:Space Mono,monospace'>{df.shape[0]} rows · {df.shape[1]} columns</div>", unsafe_allow_html=True)

        search = st.text_input("🔍", placeholder="Search all columns...", label_visibility="collapsed")

        display_df = df.copy()
        if search:
            mask = display_df.astype(str).apply(lambda r: r.str.contains(search, case=False)).any(axis=1)
            display_df = display_df[mask]

        # Custom HTML table
        num_cols_set = set(st.session_state.summary["numeric_cols"])
        rows_html = ""
        for _, row in display_df.head(50).iterrows():
            rows_html += "<tr>"
            for col in display_df.columns:
                val = row[col]
                if col in num_cols_set:
                    rows_html += f"<td class='num'>{val:,}" if isinstance(val, (int, float)) else f"<td class='num'>{val}"
                else:
                    rows_html += f"<td class='text-cell'>{val}"
                rows_html += "</td>"
            rows_html += "</tr>"

        headers = "".join(f"<th>{c}</th>" for c in display_df.columns)
        st.markdown(f"""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:16px;overflow:hidden;max-height:520px;overflow-y:auto">
            <table class="explorer-table">
                <thead><tr>{headers}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <div style="color:#6b7280;font-size:.75rem;margin-top:8px;font-family:Space Mono,monospace">
            Showing 1–{min(50,len(display_df))} of {len(display_df)}
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  PAGE: AI CHAT
# ═══════════════════════════════════════════
elif st.session_state.page == "AI Chat":
    top_bar("AI Data Chat")
    df = st.session_state.df

    st.markdown("""
    <div style="color:#6b7280;font-size:.82rem;margin-bottom:16px">
        Ask anything about your dataset
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0">
                <div class="chat-avatar" style="background:#2d1b69">🤖</div>
                <div class="chat-bubble-ai">
                    Hello! I'm your AI data analyst powered by Gemini 2.5.
                    Upload a dataset and ask me anything —
                    trends, stats, correlations, anomalies.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;align-items:flex-start;justify-content:flex-end;margin:8px 0">
                        <div class="chat-bubble-user">{msg}</div>
                        <div class="chat-avatar" style="background:#7c6ff7">U</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0">
                        <div class="chat-avatar" style="background:#2d1b69">🤖</div>
                        <div class="chat-bubble-ai">{msg}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Suggestion chips
    suggestions = [
        "Summarize the dataset", "Which column has highest values?",
        "What are the top correlations?", "Show missing data overview",
        "What trends do you see?"
    ]
    chips_html = "".join(f'<span class="chip">{s}</span>' for s in suggestions)
    st.markdown(f"<div style='margin:12px 0'>{chips_html}</div>", unsafe_allow_html=True)

    # Input
    c1, c2 = st.columns([9, 1])
    with c1:
        user_msg = st.text_input("", placeholder="Ask about your data...", label_visibility="collapsed", key="chat_input")
    with c2:
        send = st.button("SEND", use_container_width=True)

    # Chip buttons
    c = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with c[i]:
            if st.button(sug, key=f"chip_{i}", use_container_width=True):
                user_msg = sug
                send = True

    if send and user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        with st.spinner("Thinking..."):
            answer = ask_gemini(user_msg)
        st.session_state.chat_history.append(("ai", answer))
        st.rerun()

    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
