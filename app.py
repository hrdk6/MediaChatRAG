import tempfile
import os

import streamlit as st
from dotenv import load_dotenv

from pdf_loader import load_pdf
from youtube_loader import load_youtube_transcript
from vector_store import create_vector_store
from qa_chain import ask_question


st.set_page_config(
    page_title="MediaChatRAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()


# ── SESSION STATE ─────────────────────────────────────────────────────────── #

for key in ("vectorstore", "docs", "last_source", "answer", "chat_history"):
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []


# ── GLOBAL CSS ────────────────────────────────────────────────────────────── #

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">

<style>

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:          #080B10;
    --surface:     #0D1117;
    --card:        #0F1520;
    --card-hover:  #121925;
    --border:      rgba(255,255,255,0.06);
    --border-lit:  rgba(255,255,255,0.12);
    --text:        #E2E8F4;
    --text-dim:    #A0AABF;
    --muted:       #536080;
    --accent:      #5BB8FF;
    --accent2:     #9F7AEA;
    --accent3:     #38BDF8;
    --green:       #34D399;
    --yellow:      #FBBF24;
    --serif:       'Instrument Serif', Georgia, serif;
    --sans:        'DM Sans', system-ui, sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: var(--sans);
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

.block-container {
    padding: 0 3rem 5rem !important;
    max-width: 820px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-dim) !important; }

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 60px 0 52px;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(91,184,255,0.07);
    border: 1px solid rgba(91,184,255,0.18);
    border-radius: 100px;
    padding: 5px 16px 5px 11px;
    font-size: 11.5px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 32px;
}
.hero-dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: blink 2.4s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.3; transform:scale(0.65); }
}
.hero-h1 {
    font-family: var(--serif);
    font-size: clamp(42px, 5.5vw, 68px);
    font-weight: 400;
    color: var(--text);
    line-height: 1.08;
    margin: 0 0 6px;
    letter-spacing: -0.025em;
}
.hero-h1 em {
    font-style: italic;
    background: linear-gradient(115deg, var(--accent) 10%, var(--accent2) 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    margin: 18px auto 0;
    max-width: 430px;
    font-size: 16px;
    color: var(--muted);
    font-weight: 300;
    line-height: 1.65;
}

/* ── DIVIDER ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0 0 28px;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── CARD ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 24px 20px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.card-glow {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--accent) 40%,
        var(--accent2) 60%,
        transparent 100%);
    opacity: 0.6;
}

/* ── MODE TABS ── */
.mode-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
}
.mode-tab {
    flex: 1;
    padding: 11px 0;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: var(--sans);
    font-size: 13.5px;
    font-weight: 500;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
    display: flex; align-items: center; justify-content: center; gap: 7px;
}
.mode-tab:hover { border-color: var(--border-lit); color: var(--text-dim); }
.mode-tab.active {
    background: rgba(91,184,255,0.08);
    border-color: rgba(91,184,255,0.3);
    color: var(--accent);
}

/* ── STATUS PILLS ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 13px;
    border-radius: 100px;
    font-size: 12.5px;
    font-weight: 500;
    margin-bottom: 18px;
}
.pill-green {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.22);
    color: var(--green);
}
.pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
}

/* ── ANSWER ── */
.answer-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    margin-top: 18px;
}
.answer-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 13px 20px;
    border-bottom: 1px solid var(--border);
    background: rgba(91,184,255,0.04);
}
.answer-icon {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0;
}
.answer-card-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.answer-model-tag {
    margin-left: auto;
    font-size: 10.5px;
    font-weight: 500;
    color: var(--muted);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
}

/* ── HISTORY ── */
.history-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-top: 14px;
}
.hist-q {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-dim);
    display: flex;
    align-items: flex-start;
    gap: 9px;
    margin-bottom: 6px;
}
.hist-badge {
    font-size: 9.5px;
    font-weight: 700;
    background: var(--accent);
    color: #000;
    border-radius: 4px;
    padding: 2px 6px;
    flex-shrink: 0;
    margin-top: 1px;
}
.hist-a {
    font-size: 13.5px;
    color: var(--muted);
    line-height: 1.6;
    font-weight: 300;
    padding-left: 28px;
    white-space: pre-wrap;
    word-break: break-word;
}
.hist-sep {
    border: none;
    border-top: 1px solid var(--border);
    margin: 14px 0;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */

[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stFileUploader"] label {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.11em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(91,184,255,0.45) !important;
    box-shadow: 0 0 0 3px rgba(91,184,255,0.08) !important;
}
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    padding: 12px 16px !important;
    font-family: var(--sans) !important;
    font-size: 15px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(91,184,255,0.45) !important;
    box-shadow: 0 0 0 3px rgba(91,184,255,0.08) !important;
    outline: none !important;
    background: rgba(91,184,255,0.02) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    transition: all 0.18s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(91,184,255,0.35) !important;
    background: rgba(91,184,255,0.03) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--muted) !important;
    font-size: 14px !important;
}

/* Spinner text */
[data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 14px !important; }

/* Alert boxes */
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 14px !important; }
[data-testid="stSuccessAlertContent"] { color: var(--green) !important; }

/* st.markdown answer body */
.answer-md-body {
    padding: 20px 22px 24px;
    font-size: 15.5px;
    line-height: 1.82;
    color: var(--text);
    font-weight: 300;
}
.answer-md-body p  { margin: 0 0 12px; }
.answer-md-body ul,
.answer-md-body ol { padding-left: 20px; margin: 0 0 12px; }
.answer-md-body li { margin-bottom: 4px; }
.answer-md-body strong { color: #fff; font-weight: 500; }
.answer-md-body code {
    background: rgba(91,184,255,0.08);
    border: 1px solid rgba(91,184,255,0.15);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 13px;
    color: var(--accent3);
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 4px 28px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#5BB8FF,#9F7AEA);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0">✦</div>
            <span style="font-family:'Instrument Serif',serif;font-size:20px;color:#E2E8F4">MediaChatRAG</span>
        </div>
        <p style="font-size:12.5px;color:#536080;margin:0;padding-left:44px">RAG · Gemini 2.5 Flash</p>
    </div>

    <div style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#536080;margin-bottom:12px">Capabilities</div>

    <div style="display:flex;flex-direction:column;gap:3px;margin-bottom:24px">
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.025);font-size:13.5px;color:#A0AABF">
            <div style="width:6px;height:6px;border-radius:50%;background:#5BB8FF;flex-shrink:0"></div>PDF Question Answering
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.025);font-size:13.5px;color:#A0AABF">
            <div style="width:6px;height:6px;border-radius:50%;background:#F472B6;flex-shrink:0"></div>YouTube Transcript Chat
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.025);font-size:13.5px;color:#A0AABF">
            <div style="width:6px;height:6px;border-radius:50%;background:#9F7AEA;flex-shrink:0"></div>Gemini 2.5 Flash Answers
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.025);font-size:13.5px;color:#A0AABF">
            <div style="width:6px;height:6px;border-radius:50%;background:#34D399;flex-shrink:0"></div>FAISS Semantic Search
        </div>
    </div>

    <div style="font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#536080;margin-bottom:10px">Pipeline</div>
    <div style="background:rgba(91,184,255,0.04);border:1px solid rgba(91,184,255,0.1);border-radius:10px;padding:14px;font-size:12px;color:#536080;line-height:1.9">
        <span style="color:#5BB8FF">PyPDFLoader</span> / <span style="color:#F472B6">YoutubeLoader</span><br>
        → <span style="color:#9F7AEA">RecursiveCharacterTextSplitter</span><br>
        &nbsp;&nbsp;&nbsp;chunk 1000 · overlap 200<br>
        → <span style="color:#34D399">MiniLM-L6-v2</span> embeddings<br>
        → <span style="color:#FBBF24">FAISS</span> vector store<br>
        → <span style="color:#5BB8FF">Gemini 2.5 Flash</span>
    </div>
    """, unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────── #

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow"><span class="hero-dot"></span>Gemini 2.5 Flash · FAISS</div>
    <h1 class="hero-h1">Ask anything about<br>your <em>media</em></h1>
    <p class="hero-sub">Upload a PDF or paste a YouTube URL. The full pipeline — chunking, embeddings, retrieval — runs automatically.</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ── INPUT TYPE ────────────────────────────────────────────────────────────── #

st.markdown('<div class="sec-label">Source Type</div>', unsafe_allow_html=True)
option = st.selectbox("Source", ["PDF", "YouTube"], label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)


# ── PDF INPUT ─────────────────────────────────────────────────────────────── #

if option == "PDF":
    st.markdown('<div class="card"><div class="card-glow"></div><div class="sec-label">Upload PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type="pdf",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        source_key = f"pdf::{uploaded_file.name}::{uploaded_file.size}"
        if st.session_state.last_source != source_key:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                with st.spinner(f"Loading '{uploaded_file.name}' with PyPDFLoader…"):
                    st.session_state.docs = load_pdf(tmp_path)
                st.session_state.last_source = source_key
                st.session_state.vectorstore = None
                st.session_state.answer = None
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")
                st.session_state.docs = None
            finally:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ── YOUTUBE INPUT ─────────────────────────────────────────────────────────── #

elif option == "YouTube":
    st.markdown('<div class="card"><div class="card-glow"></div><div class="sec-label">YouTube URL</div>', unsafe_allow_html=True)
    youtube_url = st.text_input(
        "URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if youtube_url:
        source_key = f"yt::{youtube_url}"
        if st.session_state.last_source != source_key:
            try:
                with st.spinner("Fetching transcript via YoutubeLoader…"):
                    st.session_state.docs = load_youtube_transcript(youtube_url)
                st.session_state.last_source = source_key
                st.session_state.vectorstore = None
                st.session_state.answer = None
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Failed to fetch transcript: {e}")
                st.session_state.docs = None


# ── BUILD VECTOR STORE ────────────────────────────────────────────────────── #
# HuggingFaceEmbeddings (all-MiniLM-L6-v2) downloads the model on first run
# — this can take 30–60 s. The spinner message reflects that.

if st.session_state.docs and st.session_state.vectorstore is None:
    try:
        with st.spinner("Building FAISS index — loading MiniLM-L6-v2 embeddings (may take a moment on first run)…"):
            st.session_state.vectorstore = create_vector_store(st.session_state.docs)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")


# ── Q&A ───────────────────────────────────────────────────────────────────── #

if st.session_state.vectorstore:

    st.markdown("""
    <div class="pill pill-green">
        <div class="pill-dot" style="background:#34D399"></div>
        Knowledge base ready — ask a question
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-glow"></div><div class="sec-label">Question</div>', unsafe_allow_html=True)
    question = st.text_input(
        "Question",
        placeholder="What is this about? Summarise the key points…",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if question:
        try:
            with st.spinner("Retrieving context and querying Gemini 2.5 Flash…"):
                # ask_question returns response.content — a plain string from Gemini
                answer = ask_question(st.session_state.vectorstore, question)
            st.session_state.answer = answer
            st.session_state.chat_history.append({"q": question, "a": answer})
            if len(st.session_state.chat_history) > 6:
                st.session_state.chat_history.pop(0)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.session_state.answer = None

    # ── ANSWER ──
    # response.content is a plain string (may include markdown from Gemini).
    # Rendered with st.markdown inside a styled container — NOT via f-string
    # injection, which would be an XSS risk.
    if st.session_state.answer:
        st.markdown("""
        <div class="answer-card">
            <div class="answer-card-header">
                <div class="answer-icon">✦</div>
                <div class="answer-card-title">Answer</div>
                <div class="answer-model-tag">gemini-2.5-flash</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # st.markdown handles Gemini's markdown output safely
        with st.container():
            st.markdown(
                f'<div class="answer-md-body">',
                unsafe_allow_html=True
            )
            st.markdown(st.session_state.answer)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── HISTORY ──
    history = st.session_state.chat_history
    if len(history) > 1:
        st.markdown('<div class="history-card"><div class="sec-label">Previous Questions</div>', unsafe_allow_html=True)
        for item in reversed(history[:-1]):
            # Answers truncated for history — full text showed in answer block
            preview = item["a"][:280] + "…" if len(item["a"]) > 280 else item["a"]
            st.markdown(f"""
            <div class="hist-q">
                <span class="hist-badge">Q</span>{item['q']}
            </div>
            <div class="hist-a">{preview}</div>
            """, unsafe_allow_html=True)
            st.markdown('<hr class="hist-sep">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)