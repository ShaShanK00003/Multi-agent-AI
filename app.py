import streamlit as st
import json
import os
import uuid
import time
import base64
import logging
import traceback
from html import escape as html_escape
from datetime import datetime
from urllib.parse import urlsplit
from model_config import MODEL_CONFIGS
from history_store import load_user_data as _load_user_data, save_user_data as _save_user_data
from local_llm import DEFAULT_OLLAMA_MODEL, LOCAL_MODEL_OPTIONS, get_configured_local_model
from dotenv import set_key
from document_ingestion import DocumentIngestionError
from document_library import (
    delete_local_document,
    ingest_uploaded_document,
    retrieve_local_documents,
    supported_upload_extensions,
)
from knowledge_base import KnowledgeBase, KnowledgeBaseError
from ui_helpers import (
    build_chat_topbar_html,
    build_user_message_html,
    model_status_label,
    normalize_theme,
    set_user_theme,
    start_new_chat_state,
)
from ui_styles import build_theme_css

# ══════════════════════════════════════════════════════════════════════════════
# Page Configuration — must be FIRST Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ResearchMind · AI Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants & Helpers
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "user_data.json")
logger = logging.getLogger(__name__)

def get_logo_base64():
    logo_path = os.path.join(DATA_DIR, "logo.jpg")
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

def extract_file_content(uploaded_file):
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    text = ""
    try:
        uploaded_file.seek(0)
        if ext in [".txt", ".md", ".csv", ".json", ".py", ".html", ".js", ".ts", ".xml"]:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
            except Exception:
                try:
                    import PyPDF2
                    uploaded_file.seek(0)
                    reader = PyPDF2.PdfReader(uploaded_file)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception as ex:
                    text = f"[Could not parse PDF: {ex}]"
    except Exception as e:
        text = f"[Error reading file: {e}]"
    return text[:4000]

def extract_image_preview(uploaded_file):
    try:
        uploaded_file.seek(0)
        b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
        extension = os.path.splitext(uploaded_file.name)[1].lower()
        mime = "image/jpeg" if extension in {".jpg", ".jpeg"} else "image/png"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

# Custom friendly names → internal API model ID (cloud models only)
AVAILABLE_MODELS = {
    "Orion Pro":         {"provider": "Anthropic", "desc": "Best reasoning & nuanced writing", "badge": "Best",      "ctx": "200K", "api": "anthropic/claude-3.5-sonnet"},
    "Nova Flagship":     {"provider": "OpenAI",    "desc": "Most capable cloud model",         "badge": "Flagship",  "ctx": "128K", "api": "openai/gpt-4o"},
    "Swift Lite":        {"provider": "OpenAI",    "desc": "Fast & cost-effective",            "badge": "Fast",      "ctx": "128K", "api": "openai/gpt-4o-mini"},
    "Gemini Ultra":      {"provider": "Google",    "desc": "Multimodal & 1M context",          "badge": "Pro",       "ctx": "1M",   "api": "google/gemini-pro-1.5"},
    "DeepThink R1":      {"provider": "DeepSeek",  "desc": "Advanced step-by-step reasoning",  "badge": "Reasoning", "ctx": "64K",  "api": "deepseek/deepseek-r1"},
    # ── Offline / Local models (Ollama) ──
    "Qwen Edge 7B":      {"provider": "Offline",   "desc": "Local model · no internet needed", "badge": "Offline",   "ctx": "32K",  "api": "local:qwen2.5:7b-instruct"},
    "Qwen Edge 4B":      {"provider": "Offline",   "desc": "Ultra-light local inference",      "badge": "Offline",   "ctx": "32K",  "api": "local:qwen3:4b"},
}

PROVIDER_COLORS = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#cc785c",
    "Google":    "#4285f4",
    "Offline":   "#22c55e",
    "DeepSeek":  "#536af5",
}

# Map custom friendly names back to MODEL_CONFIGS keys for API lookups
_FRIENDLY_TO_API = {v["api"]: v["api"] for v in AVAILABLE_MODELS.values()}
# Default selection
_DEFAULT_MODEL = "Orion Pro"

SUGGESTIONS = [
    ("🔬", "Quantum computing breakthroughs 2025"),
    ("🧬", "CRISPR gene editing clinical trials progress"),
    ("⚡", "Nuclear fusion energy — commercial timeline"),
    ("🤖", "Multi-agent AI systems architecture"),
    ("🌍", "Carbon capture technology effectiveness"),
    ("💊", "mRNA technology beyond COVID vaccines"),
    ("🚀", "SpaceX Starship missions and Mars plans"),
    ("🧠", "Neuroplasticity and cognitive enhancement"),
]

RESPONSE_STYLES = ["Balanced", "Concise", "Detailed", "Academic", "Creative", "Bullet Points"]
FONT_OPTIONS    = ["Inter", "System Default", "Mono", "Serif"]
LANG_OPTIONS    = ["English", "Spanish", "French", "German", "Hindi", "Japanese", "Chinese", "Arabic"]
AVATAR_OPTIONS  = ["🧑‍💻", "👨‍🔬", "👩‍🔬", "🧑‍🎓", "👨‍💼", "👩‍💼", "🧠", "🤖", "🔬", "🎯", "💡", "⚡", "🦾", "🧬", "🚀"]


# ══════════════════════════════════════════════════════════════════════════════
# Data Layer
# ══════════════════════════════════════════════════════════════════════════════
def load_user_data():
    return _load_user_data(DATA_FILE)


def save_user_data(data):
    _save_user_data(data, DATA_FILE)


@st.cache_resource
def get_local_knowledge_base():
    """Reuse the configured persistent local database and lazy CPU embedder."""
    return KnowledgeBase()


def get_default_data():
    return {
        "profile": {
            "full_name":        "",
            "display_name":     "",
            "avatar":           "🧑‍💻",
            "work_description": "",
            "org_id":           str(uuid.uuid4())[:8].upper(),
            "memory_facts":     [],
        },
        "settings": {
            "font":           "Inter",
            "language":       "English",
            "style":          "Balanced",
            "notifications":  True,
            "show_thinking":  True,
            "local_model":    DEFAULT_OLLAMA_MODEL,
            "theme":          "Dark",
            "auto_title":     True,
        },
        "chats":    [],
        "projects": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_greeting(name):
    h = datetime.now().hour
    if   5 <= h < 12:  return f"Good morning, {name}"
    elif 12 <= h < 17: return f"Good afternoon, {name}"
    elif 17 <= h < 21: return f"Good evening, {name}"
    return f"Hello, {name}"


def get_time_group(ts_str):
    try:
        diff = datetime.now() - datetime.fromisoformat(ts_str)
        if diff.days == 0:  return "Today"
        if diff.days == 1:  return "Yesterday"
        if diff.days <= 7:  return "Last 7 Days"
        if diff.days <= 30: return "Last 30 Days"
    except Exception:
        pass
    return "Older"


def find_chat(chat_id):
    if not chat_id:
        return None
    for c in st.session_state.user_data.get("chats", []):
        if c["id"] == chat_id:
            return c
    for c in st.session_state.get("temp_chats", []):
        if c["id"] == chat_id:
            return c
    return None


def create_chat(incognito=False):
    return {
        "id":        str(uuid.uuid4()),
        "title":     "New Research",
        "messages":  [],
        "timestamp": datetime.now().isoformat(),
        "incognito": incognito,
        "pinned":    False,
        "model":     st.session_state.get("selected_model", "Claude 3.5 Sonnet"),
        "branch_of": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Session State Bootstrap
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    user_data   = load_user_data()
    if not isinstance(user_data, dict):
        user_data = get_default_data()
    has_profile = user_data.get("profile", {}).get("full_name", "").strip()
    saved_model = user_data.get("settings", {}).get("selected_model", _DEFAULT_MODEL)
    if saved_model not in AVAILABLE_MODELS:
        saved_model = _DEFAULT_MODEL
    defaults = {
        "user_data":         user_data,
        "theme":             normalize_theme(user_data.get("settings", {}).get("theme", "Dark")),
        "provider_status":   "not_checked",
        "current_view":      "welcome" if not has_profile else "home",
        "current_chat_id":   None,
        "incognito_mode":    False,
        "selected_model":    saved_model if saved_model in AVAILABLE_MODELS else _DEFAULT_MODEL,
        "pipeline_running":  False,
        "pipeline_topic":    "",
        "temp_chats":        [],
        "confirm_delete":    False,
        "canvas_open":       False,
        "canvas_content":    "",
        "canvas_title":      "",
        "comparison_mode":   False,
        "comparison_model":  "GPT-4o Mini",
        "search_query":      "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def persist_theme_change(widget_key):
    selected_theme = set_user_theme(st.session_state.user_data, st.session_state.get(widget_key))
    st.session_state.theme = selected_theme
    for other_key in ("sidebar_theme", "settings_theme"):
        if other_key != widget_key and other_key in st.session_state:
            st.session_state[other_key] = selected_theme
    save_user_data(st.session_state.user_data)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Industry-Grade Design System
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Use local system fonts; the interface does not require a third-party font request. */

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;-webkit-font-smoothing:antialiased;}

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,.stDeployButton{visibility:hidden!important;display:none!important;}
[data-testid="stHeader"],[data-testid="stDecoration"]{display:none!important;}
[data-testid="stSidebarCollapsedControl"]{visibility:visible!important;display:flex!important;background:#1a1a2e!important;border:1px solid rgba(139,92,246,0.15)!important;border-radius:8px!important;}

/* ── App shell ── */
.stApp{background:#0a0a0f!important;color:#e8e8f0!important;}
.block-container{padding:0!important;max-width:100%!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:#0f0f1a!important;border-right:1px solid rgba(139,92,246,0.1)!important;}
[data-testid="stSidebarContent"]{padding:0!important;}
[data-testid="stSidebar"] hr{border:none!important;border-top:1px solid rgba(139,92,246,0.08)!important;margin:6px 14px!important;}

/* Sidebar buttons */
[data-testid="stSidebar"] button{background:transparent!important;border:none!important;color:#9999b3!important;text-align:left!important;padding:7px 12px!important;font-size:0.83rem!important;border-radius:8px!important;font-weight:400!important;box-shadow:none!important;transition:all 0.15s ease!important;width:100%!important;font-family:'Inter',sans-serif!important;justify-content:flex-start!important;}
[data-testid="stSidebar"] button:hover{background:rgba(139,92,246,0.08)!important;color:#e0e0f5!important;}
[data-testid="stSidebar"] button[kind="primary"]{background:rgba(139,92,246,0.12)!important;color:#a78bfa!important;border:1px solid rgba(139,92,246,0.2)!important;border-radius:10px!important;font-weight:500!important;font-size:0.87rem!important;padding:9px 14px!important;}
[data-testid="stSidebar"] button[kind="primary"]:hover{background:rgba(139,92,246,0.2)!important;border-color:rgba(139,92,246,0.35)!important;color:#c4b5fd!important;}

/* ── Inputs ── */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:var(--rm-surface,#13131f)!important;border:1px solid var(--rm-border,rgba(139,92,246,0.15))!important;border-radius:10px!important;color:var(--rm-text,#e8e8f0)!important;font-family:'Inter',sans-serif!important;font-size:0.92rem!important;padding:10px 14px!important;transition:border-color 0.2s,box-shadow 0.2s!important;box-shadow:none!important;}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{border-color:rgba(139,92,246,0.5)!important;box-shadow:0 0 0 3px rgba(139,92,246,0.1)!important;outline:none!important;}
.stTextInput>label,.stTextArea>label{color:var(--rm-muted,#7070a0)!important;font-size:0.8rem!important;font-weight:500!important;}

/* ── Selectbox ── */
.stSelectbox>div>div{background:var(--rm-surface,#13131f)!important;border:1px solid var(--rm-border,rgba(139,92,246,0.15))!important;border-radius:10px!important;color:var(--rm-text,#e8e8f0)!important;}
.stSelectbox label{color:var(--rm-muted,#7070a0)!important;font-size:0.8rem!important;}
[data-baseweb="select"] span{color:var(--rm-text,#e8e8f0)!important;}

/* ── Primary buttons ── */
.block-container button[kind="primary"],.stFormSubmitButton>button{background:linear-gradient(135deg,#7c3aed,#5b21b6)!important;color:#fff!important;border:none!important;font-weight:600!important;font-size:0.9rem!important;border-radius:10px!important;padding:10px 22px!important;box-shadow:0 2px 16px rgba(124,58,237,0.25)!important;transition:all 0.2s!important;letter-spacing:0.01em!important;font-family:'Inter',sans-serif!important;}
.block-container button[kind="primary"]:hover,.stFormSubmitButton>button:hover{box-shadow:0 4px 24px rgba(124,58,237,0.4)!important;transform:translateY(-1px)!important;background:linear-gradient(135deg,#8b5cf6,#6d28d9)!important;}

/* ── Secondary buttons ── */
.block-container button[kind="secondary"]{background:rgba(139,92,246,0.08)!important;color:#a78bfa!important;border:1px solid rgba(139,92,246,0.18)!important;border-radius:8px!important;font-size:0.85rem!important;padding:8px 16px!important;font-weight:500!important;transition:all 0.15s!important;}
.block-container button[kind="secondary"]:hover{background:rgba(139,92,246,0.15)!important;border-color:rgba(139,92,246,0.3)!important;}

/* ── Chat ── */
.stChatMessage{background:transparent!important;border:none!important;padding:0!important;}

/* ── Chat input ── */
[data-testid="stChatInput"]{background:#0a0a0f!important;border-top:1px solid rgba(139,92,246,0.08)!important;padding:12px 20px!important;}
[data-testid="stChatInput"] textarea{background:var(--rm-surface,#13131f)!important;border:1px solid var(--rm-border,rgba(139,92,246,0.15))!important;border-radius:14px!important;color:var(--rm-text,#e8e8f0)!important;font-family:'Inter',sans-serif!important;font-size:0.95rem!important;padding:14px 18px!important;box-shadow:0 2px 20px rgba(0,0,0,0.3)!important;transition:border-color 0.2s,box-shadow 0.2s!important;resize:none!important;}
[data-testid="stChatInput"] textarea:focus{border-color:rgba(139,92,246,0.4)!important;box-shadow:0 0 0 3px rgba(139,92,246,0.08),0 2px 20px rgba(0,0,0,0.3)!important;}
[data-testid="stChatInput"] button{color:#8b5cf6!important;background:transparent!important;}

/* ── Status ── */
[data-testid="stStatus"]{background:#13131f!important;border:1px solid rgba(139,92,246,0.12)!important;border-radius:12px!important;font-family:'Inter',sans-serif!important;}
[data-testid="stStatus"] p,[data-testid="stStatus"] span{color:#e8e8f0!important;}
.stSpinner>div{color:#8b5cf6!important;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{gap:0!important;background:transparent!important;border-bottom:1px solid rgba(139,92,246,0.1)!important;}
.stTabs [data-baseweb="tab"]{padding:10px 20px!important;font-size:0.87rem!important;font-weight:500!important;color:#6060a0!important;border-bottom:2px solid transparent!important;background:transparent!important;transition:all 0.15s!important;}
.stTabs [aria-selected="true"]{color:#a78bfa!important;border-bottom-color:#7c3aed!important;background:transparent!important;}

/* ── Toggle ── */
.stToggle label span{color:#7070a0!important;font-size:0.85rem!important;}

/* ── Expander ── */
details{border:1px solid rgba(139,92,246,0.1)!important;border-radius:10px!important;background:#13131f!important;}
.streamlit-expanderHeader{font-size:0.85rem!important;color:#8888bb!important;font-weight:500!important;}

/* ── Download button ── */
.stDownloadButton>button{background:rgba(139,92,246,0.07)!important;color:#9090c0!important;border:1px solid rgba(139,92,246,0.12)!important;border-radius:8px!important;font-size:0.82rem!important;box-shadow:none!important;font-family:'Inter',sans-serif!important;transition:all 0.15s!important;}
.stDownloadButton>button:hover{background:rgba(139,92,246,0.14)!important;color:#c0c0e0!important;border-color:rgba(139,92,246,0.25)!important;}

/* ── Popover ── */
[data-testid="stPopover"]>div>button{background:#13131f!important;border:1px solid rgba(139,92,246,0.12)!important;border-radius:10px!important;padding:8px 14px!important;color:#c0c0e0!important;font-weight:500!important;box-shadow:none!important;transition:all 0.15s!important;font-size:0.85rem!important;font-family:'Inter',sans-serif!important;width:100%!important;}
[data-testid="stPopover"]>div>button:hover{background:rgba(139,92,246,0.08)!important;border-color:rgba(139,92,246,0.22)!important;}

/* ── Alert ── */
.stAlert{background:#13131f!important;border-color:rgba(139,92,246,0.15)!important;border-radius:10px!important;color:#e8e8f0!important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(139,92,246,0.2);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(139,92,246,0.35);}

/* ══════════════════════════════════════════
   CUSTOM COMPONENTS
   ══════════════════════════════════════════ */

/* ── Sidebar brand ── */
.sb-brand{display:flex;align-items:center;gap:10px;padding:16px 16px 12px;cursor:pointer;transition:opacity 0.2s;}
.sb-brand-icon{width:30px;height:30px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;box-shadow:0 2px 12px rgba(124,58,237,0.35);}
.sb-brand-text{font-size:0.95rem;font-weight:700;color:#e8e8f0;letter-spacing:-0.02em;}
.sb-brand-text span{color:#a78bfa;}
.sb-brand-version{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:rgba(167,139,250,0.45);font-weight:400;letter-spacing:0.06em;display:block;margin-top:1px;}
.sb-section-label{font-family:'JetBrains Mono',monospace;font-size:0.60rem;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:rgba(139,92,246,0.45);padding:8px 16px 4px;display:block;}

/* ── Incognito badge ── */
.incognito-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(139,92,246,0.1);color:#a78bfa;font-size:0.72rem;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:0.04em;border:1px solid rgba(139,92,246,0.18);text-transform:uppercase;}

/* ── Model pill ── */
.model-pill{display:inline-flex;align-items:center;gap:5px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.14);border-radius:6px;padding:3px 8px;font-size:0.75rem;color:#a78bfa;font-weight:500;white-space:nowrap;}
.model-dot{width:5px;height:5px;background:#22c55e;border-radius:50%;animation:pdot 2.5s infinite;}
@keyframes pdot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.6);}}

/* ── Welcome page ── */
.welcome-glow{position:fixed;width:800px;height:800px;background:radial-gradient(circle,rgba(124,58,237,0.06) 0%,transparent 65%);top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;z-index:0;}
.welcome-orb{width:80px;height:80px;background:linear-gradient(135deg,#7c3aed 0%,#4f46e5 100%);border-radius:22px;display:flex;align-items:center;justify-content:center;font-size:38px;margin:0 auto 28px;box-shadow:0 8px 40px rgba(124,58,237,0.4),0 0 0 1px rgba(139,92,246,0.2);animation:orbFloat 4s ease-in-out infinite;}
@keyframes orbFloat{0%,100%{transform:translateY(0px);box-shadow:0 8px 40px rgba(124,58,237,0.4);}50%{transform:translateY(-8px);box-shadow:0 16px 50px rgba(124,58,237,0.5);}}
.welcome-title{font-size:clamp(1.9rem,4.5vw,2.6rem);font-weight:800;color:#e8e8f0;text-align:center;letter-spacing:-0.035em;margin-bottom:10px;line-height:1.15;}
.welcome-title-accent{background:linear-gradient(135deg,#a78bfa,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.welcome-sub{font-size:1rem;color:rgba(232,232,240,0.5);text-align:center;margin-bottom:40px;font-weight:400;max-width:400px;line-height:1.65;margin-left:auto;margin-right:auto;}
.welcome-feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;max-width:560px;margin:0 auto 40px;}
.welcome-feat{background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.1);border-radius:12px;padding:18px 14px;text-align:center;transition:all 0.2s;}
.welcome-feat:hover{background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.2);transform:translateY(-2px);}
.welcome-feat-icon{font-size:1.6rem;margin-bottom:8px;}
.welcome-feat-title{font-size:0.8rem;font-weight:600;color:#c0c0e0;margin-bottom:3px;}
.welcome-feat-desc{font-size:0.72rem;color:rgba(232,232,240,0.35);line-height:1.4;}

/* ── Home page ── */
.home-greeting{font-size:clamp(1.7rem,3.5vw,2.3rem);font-weight:700;color:#e8e8f0;text-align:center;margin-bottom:6px;letter-spacing:-0.025em;}
.home-subtitle{font-size:1.05rem;font-weight:300;color:rgba(232,232,240,0.4);text-align:center;margin-bottom:36px;}
.home-chip-row{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:20px;}
.home-chip{display:inline-flex;align-items:center;gap:6px;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.12);border-radius:22px;padding:6px 14px;font-size:0.82rem;color:rgba(232,232,240,0.55);cursor:pointer;transition:all 0.15s;white-space:nowrap;}
.home-chip:hover{background:rgba(139,92,246,0.12);border-color:rgba(139,92,246,0.25);color:rgba(232,232,240,0.85);transform:translateY(-1px);}

/* ── Input hint ── */
.input-hint{font-size:0.72rem;color:rgba(232,232,240,0.25);text-align:center;padding:6px 0 8px;font-family:'JetBrains Mono',monospace;letter-spacing:0.02em;}

/* ── Attach chip ── */
.attach-preview{display:flex;flex-wrap:wrap;gap:6px;padding:8px 14px;}
.attach-chip{display:inline-flex;align-items:center;gap:6px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.15);border-radius:8px;padding:5px 10px;font-size:0.78rem;color:#a78bfa;}

/* ── Messages ── */
.msg-user-wrap{display:flex;justify-content:flex-end;margin:20px 0;animation:msgIn 0.22s ease-out;padding:0 20px;}
.msg-user-wrap>div{width:fit-content;max-width:68%;min-width:0;}
.msg-user{background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.15);border-radius:18px 18px 4px 18px;padding:13px 18px;max-width:100%;font-size:0.93rem;color:#e8e8f0;line-height:1.65;white-space:pre-wrap;word-break:normal;overflow-wrap:anywhere;}
.msg-user-actions{display:flex;gap:6px;justify-content:flex-end;margin-top:6px;opacity:0;transition:opacity 0.15s;}
.msg-user-wrap:hover .msg-user-actions{opacity:1;}
.msg-action-btn{background:transparent;border:none;color:rgba(232,232,240,0.3);font-size:0.7rem;cursor:pointer;padding:3px 6px;border-radius:5px;transition:all 0.12s;font-family:'Inter',sans-serif;}
.msg-action-btn:hover{color:#a78bfa;background:rgba(139,92,246,0.1);}

.msg-ai-wrap{display:flex;gap:14px;margin:20px 0;align-items:flex-start;animation:msgIn 0.22s ease-out;padding:0 20px;max-width:900px;}
@keyframes msgIn{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
.msg-ai-avatar{width:34px;height:34px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;margin-top:2px;box-shadow:0 2px 10px rgba(124,58,237,0.3);}
.msg-ai-body{flex:1;min-width:0;}
.msg-ai-name{font-size:0.78rem;font-weight:600;color:rgba(232,232,240,0.4);margin-bottom:6px;letter-spacing:0.01em;}
.msg-ai-body p{margin:0 0 10px;font-size:0.93rem;color:#e8e8f0;line-height:1.72;}
.msg-ai-body h1,.msg-ai-body h2,.msg-ai-body h3{color:#e8e8f0;margin:20px 0 8px;font-weight:700;letter-spacing:-0.01em;}
.msg-ai-body h1{font-size:1.3rem;border-bottom:1px solid rgba(139,92,246,0.1);padding-bottom:8px;}
.msg-ai-body h2{font-size:1.1rem;}
.msg-ai-body h3{font-size:0.98rem;color:#a78bfa;}
.msg-ai-body code{background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.15);border-radius:5px;padding:2px 6px;font-family:'JetBrains Mono',monospace;font-size:0.84em;color:#c4b5fd;}
.msg-ai-body pre{background:#0d0d1a;border:1px solid rgba(139,92,246,0.1);border-radius:10px;padding:16px;overflow-x:auto;font-family:'JetBrains Mono',monospace;font-size:0.84rem;margin:12px 0;}
.msg-ai-body ul,.msg-ai-body ol{padding-left:20px;margin:8px 0 12px;}
.msg-ai-body li{margin-bottom:5px;font-size:0.93rem;color:#e8e8f0;line-height:1.65;}
.msg-ai-body blockquote{border-left:3px solid #7c3aed;padding:8px 16px;margin:12px 0;background:rgba(139,92,246,0.05);border-radius:0 8px 8px 0;color:rgba(232,232,240,0.7);font-style:italic;}
.msg-ai-body table{border-collapse:collapse;width:100%;margin:12px 0;font-size:0.88rem;}
.msg-ai-body th{background:rgba(139,92,246,0.1);color:#a78bfa;padding:8px 12px;text-align:left;font-weight:600;border:1px solid rgba(139,92,246,0.12);}
.msg-ai-body td{padding:7px 12px;border:1px solid rgba(139,92,246,0.08);color:#e8e8f0;}
.msg-ai-body tr:nth-child(even) td{background:rgba(139,92,246,0.03);}
.msg-ts{font-size:0.68rem;color:rgba(232,232,240,0.25);margin-top:6px;}
.msg-actions{display:flex;gap:4px;margin-top:8px;opacity:0;transition:opacity 0.15s;}
.msg-ai-wrap:hover .msg-actions{opacity:1;}

/* ── Typing indicator ── */
.typing-wrap{display:flex;align-items:center;gap:12px;margin:16px 0;padding:0 20px;}
.typing-avatar{width:34px;height:34px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;box-shadow:0 2px 10px rgba(124,58,237,0.3);}
.typing-bubble{background:#13131f;border:1px solid rgba(139,92,246,0.12);border-radius:14px;padding:12px 18px;display:flex;gap:5px;align-items:center;}
.td{width:7px;height:7px;background:#7c3aed;border-radius:50%;animation:tdBounce 1.5s ease-in-out infinite;opacity:0.4;}
.td:nth-child(2){animation-delay:0.18s;}
.td:nth-child(3){animation-delay:0.36s;}
@keyframes tdBounce{0%,80%,100%{transform:scale(0.75);opacity:0.3;}40%{transform:scale(1.15);opacity:1;background:#a78bfa;}}

/* ── Report card ── */
.report-card{background:#13131f;border:1px solid rgba(139,92,246,0.12);border-radius:14px;padding:24px;margin:12px 0;}
.report-header{display:flex;align-items:center;gap:10px;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid rgba(139,92,246,0.08);}
.report-badge{background:rgba(124,58,237,0.12);color:#a78bfa;font-size:0.70rem;font-weight:600;padding:3px 10px;border-radius:6px;text-transform:uppercase;letter-spacing:0.07em;border:1px solid rgba(139,92,246,0.16);}

/* ── Model card ── */
.model-card{background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.1);border-radius:12px;padding:14px 16px;cursor:pointer;transition:all 0.18s;margin-bottom:8px;}
.model-card:hover{background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.22);transform:translateY(-1px);}
.model-card.selected{background:rgba(139,92,246,0.12);border-color:rgba(139,92,246,0.35);}
.model-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;}
.model-name{font-size:0.88rem;font-weight:600;color:#e8e8f0;}
.model-badge{font-size:0.65rem;font-weight:600;padding:2px 7px;border-radius:5px;letter-spacing:0.04em;text-transform:uppercase;}
.model-badge.flagship{background:rgba(251,191,36,0.12);color:#fbbf24;border:1px solid rgba(251,191,36,0.2);}
.model-badge.best{background:rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.18);}
.model-badge.fast{background:rgba(59,130,246,0.1);color:#60a5fa;border:1px solid rgba(59,130,246,0.18);}
.model-badge.pro{background:rgba(239,68,68,0.1);color:#f87171;border:1px solid rgba(239,68,68,0.18);}
.model-badge.open{background:rgba(16,185,129,0.1);color:#10b981;border:1px solid rgba(16,185,129,0.18);}
.model-badge.moe{background:rgba(245,158,11,0.1);color:#f59e0b;border:1px solid rgba(245,158,11,0.18);}
.model-badge.reasoning{background:rgba(139,92,246,0.12);color:#a78bfa;border:1px solid rgba(139,92,246,0.2);}
.model-badge.offline{background:rgba(34,197,94,0.08);color:#4ade80;border:1px solid rgba(34,197,94,0.25);}
.model-provider{font-size:0.73rem;color:rgba(232,232,240,0.4);margin-bottom:3px;}
.model-desc{font-size:0.75rem;color:rgba(232,232,240,0.35);}
.model-ctx{font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:rgba(139,92,246,0.5);margin-top:5px;}

/* ── Chat topbar ── */
.chat-topbar{display:flex;align-items:center;padding:10px 24px;border-bottom:1px solid rgba(139,92,246,0.08);gap:10px;background:rgba(10,10,15,0.95);backdrop-filter:blur(10px);flex-wrap:wrap;}
.chat-topbar-title{font-size:0.88rem;font-weight:600;color:#e8e8f0;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.topbar-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(139,92,246,0.07);border:1px solid rgba(139,92,246,0.12);border-radius:7px;padding:4px 10px;font-size:0.75rem;color:rgba(232,232,240,0.45);white-space:nowrap;}

/* ── Pipeline steps ── */
.pipeline-step{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;font-size:0.85rem;color:rgba(232,232,240,0.6);margin-bottom:4px;}
.pipeline-step.done{color:#22c55e;}
.pipeline-step.active{color:#a78bfa;background:rgba(139,92,246,0.06);}

/* ── Settings ── */
.settings-heading{font-size:1.5rem;font-weight:700;color:#e8e8f0;margin-bottom:4px;letter-spacing:-0.025em;}
.settings-sub{font-size:0.88rem;color:rgba(232,232,240,0.4);margin-bottom:28px;}
.section-label{font-family:'JetBrains Mono',monospace;font-size:0.66rem;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:#7c3aed;margin:22px 0 12px;padding-bottom:8px;border-bottom:1px solid rgba(124,58,237,0.1);display:block;}
.danger-label{font-family:'JetBrains Mono',monospace;font-size:0.66rem;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:#ef4444;margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid rgba(239,68,68,0.12);display:block;}
.info-card{background:#13131f;border:1px solid rgba(139,92,246,0.1);border-radius:10px;padding:14px 16px;margin-bottom:8px;}
.info-card-row{display:flex;align-items:center;gap:12px;}
.info-card-title{font-size:0.87rem;font-weight:500;color:#c0c0e0;}
.info-card-sub{font-size:0.73rem;color:rgba(232,232,240,0.3);margin-top:2px;}

/* ── Memory card ── */
.memory-card{background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.1);border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;gap:10px;}
.memory-fact{font-size:0.83rem;color:#c0c0e0;}

/* ── App footer ── */
.app-footer{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:rgba(232,232,240,0.12);text-align:center;padding:28px 0 12px;letter-spacing:0.06em;}
</style>
""", unsafe_allow_html=True)

# Theme-dependent design tokens are injected after the legacy shell CSS so they
# can override its fixed palette without changing the established component markup.
st.markdown(
    build_theme_css(
        st.session_state.theme,
        st.session_state.user_data.get("settings", {}).get("font", "Inter"),
    ),
    unsafe_allow_html=True,
)

# ── Hide sidebar on welcome ───────────────────────────────────────────────────
if st.session_state.current_view == "welcome":
    st.markdown("""<style>
    [data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"]{display:none!important;}
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PICKER
# ══════════════════════════════════════════════════════════════════════════════
def render_model_picker(key_prefix="mp"):
    selected = st.session_state.get("selected_model", "Claude 3.5 Sonnet")
    st.markdown("##### 🤖 Select Research Model")
    st.caption("Different models excel at different tasks. Choose wisely.")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    for name, info in AVAILABLE_MODELS.items():
        badge_cls = info["badge"].lower().replace(" ", "")
        is_sel    = name == selected
        card_cls  = "model-card selected" if is_sel else "model-card"
        prov_col  = PROVIDER_COLORS.get(info["provider"], "#8888bb")
        col_card, col_btn = st.columns([6, 1])
        with col_card:
            st.markdown(f"""
            <div class="{card_cls}">
                <div class="model-card-header">
                    <span class="model-name">{name}</span>
                    <span class="model-badge {badge_cls}">{info['badge']}</span>
                </div>
                <div class="model-provider" style="color:{prov_col}aa;">{info['provider']}</div>
                <div class="model-desc">{info['desc']}</div>
                <div class="model-ctx">Context window: {info['ctx']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            lbl = "✓ Active" if is_sel else "Select"
            if st.button(lbl, key=f"{key_prefix}_{name.replace(' ','_').replace('.','')}", use_container_width=True):
                st.session_state.selected_model = name
                st.session_state.provider_status = "not_checked"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Brand & Logo — clicking it navigates to the dashboard directly
        logo_b64 = get_logo_base64()
        logo_tag = f'<img src="data:image/jpeg;base64,{logo_b64}" style="width:34px;height:34px;border-radius:9px;object-fit:cover;box-shadow:0 0 16px rgba(124,58,237,0.5);border:1px solid rgba(167,139,250,0.3);">' if logo_b64 else '<div class="sb-brand-icon">🧠</div>'
        
        st.markdown(f"""
        <div class="sb-brand" title="ResearchMind v2.0">
            {logo_tag}
            <div>
                <div class="sb-brand-text">Research<span>Mind</span></div>
                <span class="sb-brand-version">v2.0 · AI Research Platform</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Single dashboard button — no separate "ResearchMind · Dashboard" entry
        if st.button("🏠  Dashboard", key="sb_home", use_container_width=True, help="Return to Dashboard"):
            start_new_chat_state(st.session_state)
            st.rerun()

        if st.button("✏️  New chat", key="sb_new_chat", type="primary", use_container_width=True):
            start_new_chat_state(st.session_state)
            st.rerun()

        st.session_state.setdefault("sidebar_theme", st.session_state.theme)
        st.selectbox(
            "Appearance theme",
            ["Dark", "Light"],
            key="sidebar_theme",
            label_visibility="collapsed",
            help="Switch between dark and light appearance.",
            on_change=persist_theme_change,
            args=("sidebar_theme",),
        )

        if st.button("📚  Local Document Library", key="sb_local_library", use_container_width=True):
            st.session_state.current_view = "library"
            st.rerun()

        incog = st.toggle("🛡️  Incognito mode", value=st.session_state.incognito_mode, key="sb_incog",
                          help="Chats won't be saved to history")
        if incog != st.session_state.incognito_mode:
            st.session_state.incognito_mode = incog

        st.markdown("---")

        # Search
        search_q = st.text_input("search", placeholder="🔍 Search conversations...",
                                 key="sb_search", label_visibility="collapsed")
        st.session_state.search_query = search_q.lower().strip()
        st.caption("Project grouping is not available yet.")

        # Project records are retained in user_data.json, but not surfaced as
        # controls until project-to-chat grouping has a real implementation.

        # Chat history
        st.markdown('<span class="sb-section-label">Conversations</span>', unsafe_allow_html=True)
        all_chats = st.session_state.user_data.get("chats", [])
        visible   = [c for c in all_chats if not c.get("incognito") and c.get("messages")]
        q = st.session_state.search_query
        if q:
            visible = [c for c in visible if q in c.get("title","").lower()
                       or any(q in m.get("content","").lower() for m in c.get("messages",[]))]
        visible.sort(key=lambda x: x.get("timestamp",""), reverse=True)

        if not visible:
            st.caption("No conversations yet" if not q else "No results found")
        else:
            groups = {}
            for c in visible:
                groups.setdefault(get_time_group(c.get("timestamp","")), []).append(c)
            for gname in ["Today","Yesterday","Last 7 Days","Last 30 Days","Older"]:
                if gname not in groups:
                    continue
                st.caption(gname)
                for c in groups[gname]:
                    title  = c.get("title","Untitled")[:28]
                    prefix = "📌 " if c.get("pinned") else ""
                    is_cur = c["id"] == st.session_state.current_chat_id
                    lbl    = f"{'◉ ' if is_cur else ''}💬 {prefix}{title}"
                    if st.button(lbl, key=f"sb_c_{c['id']}", use_container_width=True):
                        st.session_state.current_chat_id  = c["id"]
                        st.session_state.current_view     = "chat"
                        st.session_state.pipeline_running = False
                        st.rerun()

        st.markdown("---")

        # User account
        profile = st.session_state.user_data["profile"]
        display = profile.get("display_name") or profile.get("full_name") or "User"
        avatar  = profile.get("avatar", "🧑‍💻")
        ms = html_escape(str(st.session_state.selected_model[:36]), quote=True)
        st.markdown(f'<div class="model-pill">Model · {ms}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        with st.popover(f"{avatar}  {display}", use_container_width=True):
            st.markdown(f"**{html_escape(str(display), quote=True)}**")
            st.caption(profile.get("work_description","AI Researcher"))
            st.markdown("---")
            if st.button("⚙️  Settings",  key="sb_pop_set",  use_container_width=True):
                st.session_state.current_view = "settings"; st.rerun()
            if st.button("🧠  My Memory", key="sb_pop_mem",  use_container_width=True):
                st.session_state.current_view = "memory";   st.rerun()
            if st.button("📊  Analytics", key="sb_pop_ana",  use_container_width=True):
                st.session_state.current_view = "analytics"; st.rerun()
            st.markdown("---")
            st.caption("Single-user local profile · sign out is not available")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Welcome
# ══════════════════════════════════════════════════════════════════════════════
def render_welcome():
    st.markdown('<div class="welcome-glow"></div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        logo_b64 = get_logo_base64()
        logo_tag = f'<img src="data:image/jpeg;base64,{logo_b64}" style="width:88px;height:88px;border-radius:22px;box-shadow:0 12px 40px rgba(124,58,237,0.55),0 0 0 2px rgba(167,139,250,0.3);margin:0 auto 22px;display:block;animation:orbFloat 4s ease-in-out infinite;">' if logo_b64 else '<div class="welcome-orb">🧠</div>'
        st.markdown(f"""
        <div style="text-align:center;padding-top:6vh;">
            {logo_tag}
            <div class="welcome-title">Welcome to <span class="welcome-title-accent">ResearchMind</span></div>
            <div class="welcome-sub">The industry-grade AI research platform.<br>Powered by multi-agent intelligence.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="welcome-feat-grid">
            <div class="welcome-feat"><div class="welcome-feat-icon">🔍</div><div class="welcome-feat-title">Web Search</div><div class="welcome-feat-desc">Real-time internet research</div></div>
            <div class="welcome-feat"><div class="welcome-feat-icon">📄</div><div class="welcome-feat-title">Deep Reading</div><div class="welcome-feat-desc">Scrape & analyze sources</div></div>
            <div class="welcome-feat"><div class="welcome-feat-icon">✍️</div><div class="welcome-feat-title">AI Writing</div><div class="welcome-feat-desc">Expert report generation</div></div>
            <div class="welcome-feat"><div class="welcome-feat-icon">🧐</div><div class="welcome-feat-title">Peer Review</div><div class="welcome-feat-desc">Automatic quality check</div></div>
            <div class="welcome-feat"><div class="welcome-feat-icon">🤖</div><div class="welcome-feat-title">8 AI Models</div><div class="welcome-feat-desc">GPT, Claude, Gemini & more</div></div>
            <div class="welcome-feat"><div class="welcome-feat-icon">📊</div><div class="welcome-feat-title">Canvas View</div><div class="welcome-feat-desc">Edit reports in real-time</div></div>
        </div>
        """, unsafe_allow_html=True)

        _ROLE_OPTIONS = [
            "PhD Student",
            "ML Engineer",
            "Data Scientist",
            "Research Scientist",
            "Academic Researcher",
            "Software Engineer",
            "Product Manager",
            "Business Analyst",
            "Journalist / Writer",
            "Medical Professional",
            "Legal Professional",
            "Student (Undergrad)",
            "Student (Postgrad)",
            "Educator / Professor",
            "Entrepreneur",
            "Other",
        ]
        with st.form("welcome_form", border=False):
            name_in = st.text_input("Your name", placeholder="Enter your full name...", key="w_name")
            role_sel = st.selectbox(
                "Your role",
                options=_ROLE_OPTIONS,
                index=0,
                key="w_role_sel",
                help="Select the option that best describes you.",
            )
            submitted = st.form_submit_button("Get Started →", use_container_width=True, type="primary")

        if submitted:
            if name_in and name_in.strip():
                st.session_state.user_data["profile"]["full_name"]        = name_in.strip()
                st.session_state.user_data["profile"]["display_name"]     = name_in.strip().split()[0]
                st.session_state.user_data["profile"]["work_description"] = role_sel
                save_user_data(st.session_state.user_data)
                st.session_state.current_view = "home"
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")

        st.markdown('<div class="app-footer">ResearchMind · AI Research Platform · v2.0</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Home / Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def render_home():
    profile = st.session_state.user_data["profile"]
    display = profile.get("display_name") or profile.get("full_name") or "there"
    greeting = get_greeting(display)

    if st.session_state.incognito_mode:
        st.info("Incognito mode is on. This conversation will not be saved.")

    def start_research(topic, model, style, uploaded_files=None):
        topic = str(topic or "").strip()
        uploaded_files = uploaded_files or []
        if not topic and uploaded_files:
            topic = "Analyze the uploaded file(s): " + ", ".join(item.name for item in uploaded_files)
        if not topic:
            st.warning("Enter a research question or attach a file to continue.")
            return

        files_meta = []
        doc_context = ""
        for item in uploaded_files:
            if "image" in item.type:
                files_meta.append({
                    "name": item.name,
                    "type": item.type,
                    "preview": extract_image_preview(item),
                })
            else:
                files_meta.append({"name": item.name, "type": item.type})
                content = extract_file_content(item)
                if content:
                    doc_context += f"\n\n[Document Context from {item.name}]:\n{content}\n"

        st.session_state.selected_model = model
        st.session_state.comparison_mode = False
        new_chat = create_chat(incognito=st.session_state.incognito_mode)
        new_chat["messages"].append({
            "role": "user",
            "content": topic,
            "timestamp": datetime.now().isoformat(),
            "style": style,
            "files": files_meta,
        })
        new_chat["title"] = topic[:50]
        if st.session_state.incognito_mode:
            st.session_state.temp_chats.append(new_chat)
        else:
            st.session_state.user_data.setdefault("chats", []).append(new_chat)
            save_user_data(st.session_state.user_data)
        st.session_state.current_chat_id = new_chat["id"]
        st.session_state.current_view = "chat"
        st.session_state.pipeline_running = True
        st.session_state.pipeline_topic = topic + doc_context
        st.rerun()

    with st.container(key="home-workspace"):
        st.markdown(f"""
        <header class="home-hero">
            <div class="home-kicker">RESEARCH WORKSPACE</div>
            <h1 class="home-greeting">{html_escape(greeting, quote=True)}</h1>
            <p class="home-subtitle">What are you exploring today?</p>
        </header>
        """, unsafe_allow_html=True)

        st.markdown('<div class="home-suggestions-label">Try a research prompt</div>', unsafe_allow_html=True)
        suggestion_cols = st.columns(3, gap="small")
        for index, (emoji, prompt) in enumerate(SUGGESTIONS[:3]):
            with suggestion_cols[index]:
                if st.button(
                    f"{emoji}  {prompt}",
                    key=f"sug_btn_{index}",
                    use_container_width=True,
                    help="Start research using this question.",
                ):
                    start_research(
                        prompt,
                        st.session_state.selected_model,
                        st.session_state.user_data.get("settings", {}).get("style", "Balanced"),
                    )

        with st.form("home_form", clear_on_submit=True, border=True):
            topic = st.text_area(
                "Research question",
                placeholder="Ask a question, explore a topic, or analyze an attached document…",
                key="home_topic_in",
                label_visibility="collapsed",
                height=76,
            )
            # ── Compact upload row ──
            up_col, _ = st.columns([1, 5])
            with up_col:
                uploaded = st.file_uploader(
                    "📎",
                    type=["pdf", "txt", "md", "png", "jpg", "jpeg", "csv", "docx"],
                    accept_multiple_files=True,
                    key="home_file_up",
                    label_visibility="collapsed",
                    help="Upload files · 200 MB per file · PDF, TXT, MD, PNG, JPG, CSV, DOCX",
                )
            controls = st.columns([1.5, 1.25, 1.1], vertical_alignment="bottom")
            with controls[0]:
                model_keys = list(AVAILABLE_MODELS)
                current_model = st.session_state.selected_model
                model_index = model_keys.index(current_model) if current_model in model_keys else 0
                selected_model = st.selectbox("Model", model_keys, index=model_index, key="home_model_sel")
            with controls[1]:
                styles = st.session_state.user_data.get("settings", {}).get("style", "Balanced")
                style_index = RESPONSE_STYLES.index(styles) if styles in RESPONSE_STYLES else 0
                selected_style = st.selectbox(
                    "Response style", RESPONSE_STYLES, index=style_index, key="home_style_sel",
                    help="Choose how the AI structures its response.",
                )
            with controls[2]:
                submitted = st.form_submit_button("Start research", type="primary", use_container_width=True)

    if submitted:
        start_research(topic, selected_model, selected_style, uploaded)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Chat
# ══════════════════════════════════════════════════════════════════════════════
def render_chat():
    chat = find_chat(st.session_state.current_chat_id)
    if not chat:
        st.session_state.current_view = "home"
        st.rerun()
        return

    model_info     = AVAILABLE_MODELS.get(st.session_state.selected_model, {})
    model_provider = model_info.get("provider", "AI")

    # ── Top bar ──
    show_incognito_badge = bool(chat.get("incognito") or st.session_state.incognito_mode)
    last_response_model = next(
        (
            message.get("model")
            for message in reversed(chat.get("messages", []))
            if message.get("role") == "assistant"
            and message.get("type") in {"research", "local_chat"}
            and message.get("model")
        ),
        None,
    )

    tb1, tb2, tb3 = st.columns([6, 1.2, 1.2])
    with tb1:
        st.markdown(
            build_chat_topbar_html(
                chat.get("title", "Research Session"),
                st.session_state.selected_model,
                last_response_model=last_response_model,
                incognito=show_incognito_badge,
            ),
            unsafe_allow_html=True,
        )
    with tb2:
        if st.button("📋 Canvas", key="tb_canvas", use_container_width=True, type="secondary"):
            st.session_state.canvas_open = not st.session_state.canvas_open
            st.rerun()
    with tb3:
        last_report = None
        for m in reversed(chat.get("messages", [])):
            if m.get("type") == "research" and m.get("data", {}).get("report"):
                last_report = m["data"]["report"]
                break
        if last_report:
            st.download_button("⬇ Export", data=last_report,
                               file_name=f"research_{chat['id'][:8]}.md",
                               mime="text/markdown", key="tb_dl", use_container_width=True)

    # ── Message area + optional canvas split ──
    if st.session_state.canvas_open:
        msg_col, canvas_col = st.columns([1, 1])
    else:
        msg_col = st.container()
        canvas_col = None

    with msg_col:
        for i, msg in enumerate(chat.get("messages", [])):
            if msg["role"] == "user":
                try:
                    ts_fmt = datetime.fromisoformat(msg.get("timestamp","")).strftime("%I:%M %p")
                except Exception:
                    ts_fmt = ""
                st.markdown(
                    build_user_message_html(msg.get("content", ""), msg.get("files"), ts_fmt),
                    unsafe_allow_html=True,
                )

                col_ua1, col_ua2 = st.columns([5, 1])
                with col_ua2:
                    with st.popover("✏️ Edit", key=f"edit_pop_{i}"):
                        st.markdown("<span style='font-size:0.85rem;font-weight:600;color:#e8e8f0;'>Revise Prompt</span>", unsafe_allow_html=True)
                        revised_text = st.text_area("Prompt", value=msg['content'], key=f"rev_prompt_{i}", height=100, label_visibility="collapsed")
                        if st.button("Re-run ↵", key=f"rev_run_{i}", type="primary", use_container_width=True):
                            msg['content'] = revised_text
                            chat["messages"] = chat["messages"][:i+1]
                            st.session_state.pipeline_running = True
                            st.session_state.pipeline_topic = revised_text
                            if not chat.get("incognito"):
                                save_user_data(st.session_state.user_data)
                            st.rerun()

            elif msg["role"] == "assistant":
                response_model = msg.get("model") or st.session_state.selected_model
                response_provider = (
                    response_model
                    if isinstance(response_model, str) and response_model.endswith("(local)")
                    else AVAILABLE_MODELS.get(response_model, {}).get("provider", model_provider)
                )
                response_provider_html = html_escape(str(response_provider), quote=True)
                try:
                    ts_fmt = datetime.fromisoformat(msg.get("timestamp","")).strftime("%I:%M %p")
                except Exception:
                    ts_fmt = ""

                if msg.get("type") == "local_chat":
                    data = msg.get("data", {})
                    st.markdown(f"""
                    <div class="msg-ai-wrap">
                        <div class="msg-ai-avatar">🧠</div>
                        <div class="msg-ai-body">
                            <div class="msg-ai-name">ResearchMind · {response_provider_html} · {ts_fmt}</div>
                    """, unsafe_allow_html=True)
                    st.markdown(data.get("report", ""))
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    local_sources = data.get("sources") or []
                    if local_sources:
                        with st.expander("📚 Local document sources", expanded=False):
                            for source in local_sources:
                                location = f" · page {source['page_number']}" if source.get("page_number") is not None else ""
                                st.caption(f"[{source.get('source_id', 'source unavailable')}] {source.get('filename') or source.get('title') or 'Local document'}{location}")
                elif msg.get("type") == "research":
                    data = msg.get("data", {})
                    st.markdown(f"""
                    <div class="msg-ai-wrap">
                        <div class="msg-ai-avatar">🧠</div>
                        <div class="msg-ai-body">
                            <div class="msg-ai-name">ResearchMind · {response_provider_html} · {ts_fmt}</div>
                    """, unsafe_allow_html=True)

                    if data.get("report"):
                        st.markdown("""<div class="report-card"><div class="report-header">
                        <span class="report-badge">📊 Research Report</span>
                        </div>""", unsafe_allow_html=True)
                        st.markdown(data["report"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    if data.get("critic"):
                        with st.expander("🧐 AI Peer Review", expanded=True):
                            st.markdown(data["critic"])

                    st.caption(
                        "Source IDs are checked against sources returned by search or local retrieval. "
                        "This does not verify that each claim is supported by its citation; unfetched "
                        "search results provide snippets only, and uncited statements may be model summaries."
                    )
                    research_sources = data.get("sources") or []
                    if research_sources:
                        with st.expander("🔗 Research sources and fetch status", expanded=False):
                            for source in research_sources:
                                source_id = str(source.get("source_id") or "source unavailable")
                                title = str(source.get("title") or source.get("filename") or "Source")
                                if source.get("source_type") == "web":
                                    url = source.get("url") or source.get("search_url")
                                    status = "Page fetched" if source.get("fetched") else "Search result only; page not fetched"
                                    st.text(f"[{source_id}] {title} · {status}")
                                    if url and urlsplit(str(url)).scheme.lower() in {"http", "https"}:
                                        st.link_button("Open source", url, key=f"source_{i}_{source_id[:12]}")
                                else:
                                    location = f" · page {source['page_number']}" if source.get("page_number") is not None else ""
                                    st.caption(f"[{source_id}] {title}{location} · local document")

                    # Action bar
                    ac1, ac2, ac3, ac4, ac5 = st.columns(5)
                    with ac1:
                        if data.get("report"):
                            st.download_button("⬇ .md", data=data["report"],
                                               file_name=f"research_{chat['id'][:8]}.md",
                                               mime="text/markdown", key=f"dl_md_{i}")
                    with ac2:
                        if data.get("report"):
                            st.download_button("⬇ .txt", data=data["report"],
                                               file_name=f"research_{chat['id'][:8]}.txt",
                                               mime="text/plain", key=f"dl_txt_{i}")
                    with ac3:
                        if data.get("report") and st.button("📋 Canvas", key=f"open_canvas_{i}", type="secondary"):
                            st.session_state.canvas_content = data["report"]
                            st.session_state.canvas_title   = chat.get("title","Research Report")
                            st.session_state.canvas_open    = True
                            st.rerun()
                    with ac4:
                        if st.button("🔄 Regen", key=f"regen_{i}", type="secondary"):
                            if i > 0:
                                prev = chat["messages"][i-1]
                                st.session_state.pipeline_running = True
                                st.session_state.pipeline_topic   = prev.get("content","")
                                chat["messages"].pop(i)
                                if not chat.get("incognito"):
                                    save_user_data(st.session_state.user_data)
                                st.rerun()
                    with ac5:
                        if st.button("⎇ Branch", key=f"branch_{i}", type="secondary"):
                            new_branch = create_chat(incognito=chat.get("incognito", False))
                            new_branch["messages"] = chat["messages"][:i+1].copy()
                            new_branch["title"]    = f"Branch: {chat.get('title','')[:28]}"
                            new_branch["branch_of"]= chat["id"]
                            if not chat.get("incognito"):
                                st.session_state.user_data.setdefault("chats",[]).append(new_branch)
                                save_user_data(st.session_state.user_data)
                            else:
                                st.session_state.temp_chats.append(new_branch)
                            st.session_state.current_chat_id = new_branch["id"]
                            st.rerun()

                    st.markdown("</div></div>", unsafe_allow_html=True)

                elif msg.get("type") == "error":
                    st.error(f"⚠️ {msg.get('content','An error occurred.')}")
                else:
                    st.markdown(f"""
                    <div class="msg-ai-wrap">
                        <div class="msg-ai-avatar">🧠</div>
                        <div class="msg-ai-body">
                            <div class="msg-ai-name">ResearchMind · {ts_fmt}</div>
                            <div>{html_escape(str(msg.get('content','')), quote=True)}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Pipeline execution ──────────────────────────────────────────────
        if st.session_state.pipeline_running:
            topic     = st.session_state.pipeline_topic
            model_api = AVAILABLE_MODELS.get(st.session_state.selected_model, {}).get("api", MODEL_CONFIGS["GPT-4o Mini"])

            st.markdown("""
            <div class="typing-wrap">
                <div class="typing-avatar">🧠</div>
                <div class="typing-bubble"><div class="td"></div><div class="td"></div><div class="td"></div></div>
                <span style="font-size:0.78rem;color:rgba(232,232,240,0.3);">ResearchMind is researching...</span>
            </div>
            """, unsafe_allow_html=True)

            safe_error_types = ()
            try:
                from chat_service import ChatGenerationError, run_research_with_local_fallback
                from research_pipeline import sources_for_history
                safe_error_types = (ChatGenerationError,)

                with st.status("🔬 Running Research Pipeline...", expanded=True) as status:
                    st.session_state.provider_status = "not_checked"
                    st.markdown('<div class="pipeline-step active">🔍 Preparing research and generation...</div>', unsafe_allow_html=True)
                    prior_messages = chat.get("messages", [])[:-1]
                    research_result = run_research_with_local_fallback(
                        topic,
                        model_api,
                        history=prior_messages,
                        local_model=get_configured_local_model(
                            st.session_state.user_data.get("settings")
                        ),
                    )
                    search_results = research_result["search_results"]
                    reader_results = research_result["scraped_content"]
                    report = research_result["report"]
                    critic = research_result["feedback"]
                    if research_result.get("generation_mode") == "local":
                        fallback_notice = (research_result.get("fallback_notice") or "").lower()
                        if "requested project file" in fallback_notice:
                            st.session_state.provider_status = "local_file"
                        elif "not configured" in fallback_notice:
                            st.session_state.provider_status = "api_unconfigured"
                        else:
                            st.session_state.provider_status = "api_fallback"
                        st.info(research_result["fallback_notice"])
                        st.markdown('<div class="pipeline-step done">🖥️ Answer generated locally; web search and peer review were skipped.</div>', unsafe_allow_html=True)
                        status.update(label="✅ Local Answer Ready", state="complete")
                    else:
                        st.session_state.provider_status = "api_success"
                        st.markdown('<div class="pipeline-step done">✅ Web search complete — sources tracked</div>', unsafe_allow_html=True)
                        st.markdown('<div class="pipeline-step done">✅ Source reading complete — fetch status recorded</div>', unsafe_allow_html=True)
                        st.markdown('<div class="pipeline-step done">✅ Report drafted — structured and formatted</div>', unsafe_allow_html=True)
                        st.markdown('<div class="pipeline-step active">🧐 Running AI peer review...</div>', unsafe_allow_html=True)
                        st.markdown('<div class="pipeline-step done">✅ Peer review complete — quality assessed</div>', unsafe_allow_html=True)
                        status.update(label="✅ Research Complete!", state="complete")

                # Auto-populate canvas
                st.session_state.canvas_content = report
                st.session_state.canvas_title   = topic[:50]

                chat["messages"].append({
                    "role":      "assistant",
                    "type":      research_result.get("message_type", "research"),
                    "data":      {"search": search_results, "reader": reader_results, "report": report, "critic": critic,
                                  "sources": sources_for_history(research_result["sources"]), "search_error": research_result["search_error"],
                                  "local_retrieval_error": research_result.get("local_retrieval_error"),
                                  "citation_validation": research_result.get("citation_validation"),
                                  "topic_truncated": research_result.get("topic_truncated", False),
                                  "generation_mode": research_result.get("generation_mode", "api_research"),
                                  "fallback_notice": research_result.get("fallback_notice")},
                    "timestamp": datetime.now().isoformat(),
                    "model":     research_result.get("model_label", st.session_state.selected_model),
                })
                if chat.get("title") == "New Research":
                    chat["title"] = topic[:50]

            except Exception as e:
                st.session_state.provider_status = "generation_error"
                if isinstance(e, safe_error_types):
                    logger.warning("Research request failed (%s).", type(e).__name__)
                    public_error = str(e)
                else:
                    # Keep traceback frames for debugging but omit the exception value,
                    # which can contain provider details. Never log environment values.
                    stack = "".join(traceback.format_tb(e.__traceback__))
                    logger.error(
                        "Unexpected research failure (%s). Traceback frames:\n%s",
                        type(e).__name__,
                        stack,
                    )
                    public_error = "An unexpected internal error occurred. Please retry; details were recorded in the development log."
                st.error(f"❌ Pipeline error: {public_error}")
                chat["messages"].append({
                    "role":      "assistant",
                    "type":      "error",
                    "content":   f"The research pipeline encountered an error: {public_error}",
                    "timestamp": datetime.now().isoformat(),
                })

            st.session_state.pipeline_running = False
            if not chat.get("incognito", False):
                save_user_data(st.session_state.user_data)
            st.rerun()

        # ── Follow-up input ─────────────────────────────────────────────────
        fu_file = st.file_uploader("Attach file", type=["pdf","txt","md","png","jpg","csv","docx"],
                                   key=f"fu_{chat['id']}", label_visibility="collapsed")
        if prompt := st.chat_input("Ask a follow-up question or start a new research thread..."):
            fu_meta = []
            fu_doc_context = ""
            if fu_file:
                if "image" in fu_file.type:
                    b64_prev = extract_image_preview(fu_file)
                    fu_meta.append({"name": fu_file.name, "type": fu_file.type, "preview": b64_prev})
                else:
                    dtext = extract_file_content(fu_file)
                    fu_meta.append({"name": fu_file.name, "type": fu_file.type})
                    if dtext:
                        fu_doc_context += f"\n\n[Document Context from {fu_file.name}]:\n{dtext}\n"

            fu_pipeline_topic = prompt + (fu_doc_context if fu_doc_context else "")

            chat["messages"].append({
                "role":      "user",
                "content":   prompt,
                "timestamp": datetime.now().isoformat(),
                "files":     fu_meta,
            })
            if chat.get("title") == "New Research":
                chat["title"] = prompt[:50]
            st.session_state.pipeline_running = True
            st.session_state.pipeline_topic   = fu_pipeline_topic
            if not chat.get("incognito"):
                save_user_data(st.session_state.user_data)
            st.rerun()

    # ── Canvas panel ────────────────────────────────────────────────────────
    if st.session_state.canvas_open and canvas_col is not None:
        with canvas_col:
            st.markdown("""
            <div style="padding:14px 18px;border-bottom:1px solid rgba(139,92,246,0.1);background:rgba(139,92,246,0.03);">
                <span style="font-size:0.88rem;font-weight:600;color:#e8e8f0;">📋 Canvas — Artifact Viewer</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✕ Close Canvas", key="close_canvas", type="secondary"):
                st.session_state.canvas_open = False
                st.rerun()
            if st.session_state.canvas_content:
                st.markdown(f"**{st.session_state.canvas_title}**")
                ct1, ct2 = st.tabs(["📄 Rendered", "📝 Markdown"])
                with ct1:
                    st.markdown(st.session_state.canvas_content)
                with ct2:
                    edited = st.text_area("Edit report", value=st.session_state.canvas_content,
                                          height=500, key="canvas_edit", label_visibility="collapsed")
                    if st.button("💾 Save edits", key="canvas_save", type="primary"):
                        st.session_state.canvas_content = edited
                        st.success("Saved!")
                st.download_button("⬇ Download Report", data=st.session_state.canvas_content,
                                   file_name=f"{st.session_state.canvas_title[:30]}.md",
                                   mime="text/markdown", key="canvas_dl")
            else:
                st.caption("Canvas is empty. Run a research session to populate it automatically.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Local Document Library
# ══════════════════════════════════════════════════════════════════════════════
def render_local_library():
    st.title("📚 Local Document Library")
    st.caption("Index and search your documents locally. This page does not call web search or external language-model providers.")

    try:
        knowledge_base = get_local_knowledge_base()
    except KnowledgeBaseError as exc:
        st.error(f"Local knowledge base is unavailable: {exc}")
        return
    except Exception as exc:
        st.error(f"Local knowledge base could not be initialized ({type(exc).__name__}).")
        return

    with st.container():
        st.subheader("Add documents")
        st.caption("Supported: PDF, TXT, Markdown, and DOCX; maximum file size is 20 MiB. Files are indexed only after you select Index document.")
        upload = st.file_uploader(
            "Choose a document", type=supported_upload_extensions(), key="library_upload"
        )
        if upload is not None and st.button("Index document", key="library_index", type="primary"):
            try:
                with st.spinner("Extracting text and indexing with the local embedding model…"):
                    result = ingest_uploaded_document(knowledge_base, upload.name, upload.getvalue())
                if result.get("duplicate"):
                    st.info(f"Already indexed: {result.get('source_filename', upload.name)}. No new chunks were added.")
                else:
                    st.success(
                        f"Indexed {result.get('source_filename', upload.name)} "
                        f"({result.get('chunk_count', 0)} chunks)."
                    )
            except DocumentIngestionError as exc:
                st.error(str(exc))
            except KnowledgeBaseError as exc:
                st.error(f"Indexing failed: {exc}")
            except Exception as exc:
                st.error(f"Indexing failed ({type(exc).__name__}). Check the local model and database, then try again.")

    st.markdown("### Search local documents")
    query = st.text_input("Search indexed documents", key="library_query", placeholder="Ask a question about your indexed files")
    if query.strip():
        try:
            records = retrieve_local_documents(knowledge_base, query, top_k=5)
            if not records:
                st.info("No matching local passages were found.")
            for index, record in enumerate(records, start=1):
                source = record.get("source") or "Unknown source"
                page = record.get("page")
                location = f" · page {page}" if page is not None else ""
                similarity = record.get("similarity")
                score = f" · similarity {similarity:.3f}" if isinstance(similarity, (int, float)) else ""
                with st.expander(f"{index}. {source}{location}{score}"):
                    st.text(record.get("text") or "")
                    st.caption(f"Document ID: {record.get('document_id') or 'unavailable'}")
                    st.caption(f"Chunk ID: {record.get('chunk_id') or 'unavailable'}")
        except KnowledgeBaseError as exc:
            st.error(f"Local retrieval failed: {exc}")
        except Exception as exc:
            st.error(f"Local retrieval failed ({type(exc).__name__}). Check the local model and database.")

    st.markdown("### Indexed documents")
    try:
        documents = knowledge_base.list_documents()
    except KnowledgeBaseError as exc:
        st.error(f"Could not list indexed documents: {exc}")
        return
    except Exception as exc:
        st.error(f"Could not list indexed documents ({type(exc).__name__}).")
        return

    if not documents:
        st.caption("No indexed documents were found in the configured local knowledge base.")
        return

    for document in documents:
        label = f"{document['source_filename']} · {document['file_type'].upper()} · {document['chunk_count']} chunks"
        with st.expander(label):
            st.caption(f"Document ID: {document['document_id']}")
            if st.button("Delete this document…", key=f"library_delete_select_{document['document_id']}"):
                st.session_state.library_delete_confirmation = document["document_id"]
            if st.session_state.get("library_delete_confirmation") == document["document_id"]:
                st.warning(f"Delete {document['source_filename']} and its indexed chunks? This cannot be undone.")
                confirm = st.checkbox("I confirm deletion of this document", key=f"library_delete_confirm_{document['document_id']}")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Confirm delete", key=f"library_delete_{document['document_id']}", disabled=not confirm, type="primary"):
                        try:
                            outcome = delete_local_document(
                                knowledge_base, document["document_id"], confirmed=confirm
                            )
                            if outcome and outcome.get("deleted"):
                                st.session_state.library_delete_confirmation = None
                                st.success(f"Deleted {outcome['deleted_chunks']} indexed chunks.")
                                st.rerun()
                            st.info("No indexed chunks matched this document.")
                        except KnowledgeBaseError as exc:
                            st.error(f"Deletion failed: {exc}")
                        except Exception as exc:
                            st.error(f"Deletion failed ({type(exc).__name__}).")
                with cancel_col:
                    if st.button("Cancel", key=f"library_cancel_delete_{document['document_id']}"):
                        st.session_state.library_delete_confirmation = None
                        st.rerun()

    if st.button("← Back to Dashboard", key="library_back"):
        st.session_state.current_view = "home"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Settings
# ══════════════════════════════════════════════════════════════════════════════
def render_settings():
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown("""
        <div class="settings-heading">Settings</div>
        <div class="settings-sub">Customize your ResearchMind experience</div>
        """, unsafe_allow_html=True)

        tab_gen, tab_acc, tab_models, tab_api = st.tabs(["⚙️ General", "👤 Profile", "🤖 Models", "🔑 API"])
        settings = st.session_state.user_data["settings"]
        profile  = st.session_state.user_data["profile"]

        with tab_gen:
            st.markdown('<span class="section-label">Preferences</span>', unsafe_allow_html=True)
            st.session_state.setdefault("settings_theme", st.session_state.theme)
            st.selectbox(
                "Appearance theme",
                ["Dark", "Light"],
                key="settings_theme",
                help="Choose the color theme for this local application.",
                on_change=persist_theme_change,
                args=("settings_theme",),
            )
            new_font  = st.selectbox("Interface font", FONT_OPTIONS,
                                     index=FONT_OPTIONS.index(settings.get("font","Inter")) if settings.get("font") in FONT_OPTIONS else 0,
                                     key="set_font", help="Applied to the interface when settings are saved.")
            new_style = st.selectbox("Default response style", RESPONSE_STYLES,
                                     index=RESPONSE_STYLES.index(settings.get("style","Balanced")) if settings.get("style") in RESPONSE_STYLES else 0,
                                     key="set_style", help="Saved as a preference; style prompting is not wired yet.")
            new_lang  = st.selectbox("Language", LANG_OPTIONS,
                                     index=LANG_OPTIONS.index(settings.get("language","English")) if settings.get("language") in LANG_OPTIONS else 0,
                                     key="set_lang", help="Saved as a preference; output translation is not wired yet.")
            st.markdown('<span class="section-label">AI Behavior</span>', unsafe_allow_html=True)
            st.caption("Only the final model response is shown. Automatic notifications and configurable auto-titles are not available yet.")
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("💾 Save Settings", key="save_gen", type="primary"):
                settings.update({"font":new_font,"style":new_style,"language":new_lang})
                if "home_style_sel" in st.session_state:
                    st.session_state.home_style_sel = new_style
                save_user_data(st.session_state.user_data)
                st.success("✅ Settings saved!")
                time.sleep(0.4); st.rerun()

        with tab_acc:
            st.markdown('<span class="section-label">Identity</span>', unsafe_allow_html=True)
            cur_av = profile.get("avatar","🧑‍💻")
            av_idx = AVATAR_OPTIONS.index(cur_av) if cur_av in AVATAR_OPTIONS else 0
            st.markdown(f'<div style="font-size:3.5rem;margin:4px 0 12px;">{html_escape(str(cur_av), quote=True)}</div>', unsafe_allow_html=True)
            new_av      = st.selectbox("Avatar", AVATAR_OPTIONS, index=av_idx, key="set_av", format_func=lambda x: f"{x} Avatar")
            new_name    = st.text_input("Full name",               value=profile.get("full_name",""),        key="set_name")
            new_display = st.text_input("Display name (nickname)", value=profile.get("display_name",""),     key="set_display", placeholder="e.g. Alex")
            new_work    = st.text_input("Your role / work",        value=profile.get("work_description",""), key="set_work",    placeholder="e.g. AI Researcher, PhD Student")
            st.markdown('<span class="section-label">Account Info</span>', unsafe_allow_html=True)
            st.text_input("Organization ID", value=profile.get("org_id","N/A"), disabled=True, key="set_org")
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-row">
                    <span style="font-size:1.2rem;">🌐</span>
                    <div><div class="info-card-title">Session Started</div>
                    <div class="info-card-sub">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("💾 Save Profile", key="save_prof", type="primary"):
                profile.update({"avatar":new_av,"full_name":new_name,"display_name":new_display,"work_description":new_work})
                save_user_data(st.session_state.user_data)
                st.success("✅ Profile updated!")
                time.sleep(0.4); st.rerun()
            st.markdown('<span class="danger-label">Danger Zone</span>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                if st.button("🧹 Clear history", key="clr_hist", use_container_width=True):
                    st.session_state.user_data["chats"] = []
                    save_user_data(st.session_state.user_data)
                    st.success("✅ History cleared!"); time.sleep(0.3); st.rerun()
            with d2:
                if st.button("🗑️ Delete local profile and history", key="del_acct", use_container_width=True):
                    st.session_state.confirm_delete = True
            if st.session_state.get("confirm_delete"):
                st.warning("This removes the local profile and saved conversation history. Indexed knowledge-base documents are unchanged.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, delete everything", key="yes_del", type="primary"):
                        if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
                        for k in list(st.session_state.keys()): del st.session_state[k]
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="no_del"):
                        st.session_state.confirm_delete = False; st.rerun()

        with tab_models:
            st.markdown('<span class="section-label">Available Models</span>', unsafe_allow_html=True)
            st.caption("Select your preferred model. Different models excel at different tasks.")
            render_model_picker(key_prefix="smp")
            st.markdown('<span class="section-label">Default Model</span>', unsafe_allow_html=True)
            model_keys = list(AVAILABLE_MODELS.keys())
            cur_idx    = model_keys.index(st.session_state.selected_model) if st.session_state.selected_model in model_keys else 0
            new_def    = st.selectbox("Set default model", model_keys, index=cur_idx, key="set_def_model")
            if st.button("Set as Default", key="save_def_m", type="primary"):
                st.session_state.selected_model = new_def
                st.session_state.provider_status = "not_checked"
                st.session_state.user_data.setdefault("settings", {})["selected_model"] = new_def
                save_user_data(st.session_state.user_data)
                st.success(f"✅ Default model set to {new_def}")

            st.markdown('<span class="section-label">Local fallback model</span>', unsafe_allow_html=True)
            st.caption(f"Preferred local fallback: {DEFAULT_OLLAMA_MODEL}. Qwen3 4B remains available as a rollback option.")
            st.caption("Qwen3.5 4B is experimental: it passed several format checks but gave inconsistent results on repeated arithmetic tests.")
            local_model_ids = list(LOCAL_MODEL_OPTIONS.values())
            current_local_model = get_configured_local_model(
                st.session_state.user_data.get("settings")
            )
            selected_local_model = st.selectbox(
                "Local Ollama model",
                local_model_ids,
                index=local_model_ids.index(current_local_model),
                key="local_fallback_model_sel",
                help="Select an installed local model. This does not change your OpenRouter model preference.",
            )
            if st.button("Save local fallback model", key="save_local_fallback_model"):
                st.session_state.user_data.setdefault("settings", {})["local_model"] = selected_local_model
                save_user_data(st.session_state.user_data)
                st.success(f"Local fallback model set to {selected_local_model}")

        with tab_api:
            st.markdown('<span class="section-label">API Keys</span>', unsafe_allow_html=True)
            st.caption("Keys are stored only in your local .env file and never sent externally.")
            st.markdown("""
            <div class="info-card"><div class="info-card-row"><span style="font-size:1.2rem;">🔑</span>
            <div><div class="info-card-title">OpenRouter API Key</div>
            <div class="info-card-sub">Access to GPT-4o, Claude, Gemini, Llama, Mixtral, DeepSeek</div></div></div></div>
            <div class="info-card"><div class="info-card-row"><span style="font-size:1.2rem;">🌐</span>
            <div><div class="info-card-title">Tavily Search API Key</div>
            <div class="info-card-sub">Powers the real-time web search agent</div></div></div></div>
            """, unsafe_allow_html=True)
            st.markdown('<span class="section-label">Update Keys</span>', unsafe_allow_html=True)
            or_key  = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...", key="api_or")
            tav_key = st.text_input("Tavily API Key",     type="password", placeholder="tvly-...",     key="api_tav")
            if st.button("💾 Save API Keys", key="save_api_k", type="primary"):
                env_path = os.path.join(DATA_DIR, ".env")
                try:
                    if or_key:
                        set_key(env_path, "OPENROUTER_API_KEY", or_key, quote_mode="always")
                    if tav_key:
                        set_key(env_path, "TAVILY_API_KEY", tav_key, quote_mode="always")
                    st.success("✅ API keys saved to .env file!")
                except Exception as e:
                    st.error(f"Could not update .env ({type(e).__name__}). Check file permissions and try again.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Memory
# ══════════════════════════════════════════════════════════════════════════════
def render_memory():
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown("""
        <div class="settings-heading">🧠 Memory</div>
        <div class="settings-sub">Manage private notes stored on this device. Notes are not automatically sent to the model.</div>
        """, unsafe_allow_html=True)
        profile = st.session_state.user_data["profile"]
        facts   = profile.get("memory_facts", [])
        st.markdown('<span class="section-label">Saved Facts</span>', unsafe_allow_html=True)
        if not facts:
            st.info("No memory facts yet. Add some below to personalize your experience.")
        else:
            for i, fact in enumerate(facts):
                fc, fd = st.columns([5, 1])
                with fc:
                    st.markdown(f'<div class="memory-card"><span class="memory-fact">💡 {html_escape(str(fact), quote=True)}</span></div>', unsafe_allow_html=True)
                with fd:
                    if st.button("✕", key=f"del_mem_{i}", help="Delete this saved note"):
                        facts.pop(i)
                        profile["memory_facts"] = facts
                        save_user_data(st.session_state.user_data)
                        st.rerun()
        st.markdown('<span class="section-label">Add Memory</span>', unsafe_allow_html=True)
        with st.form("add_mem_f", border=False):
            new_fact = st.text_input("New fact", placeholder="e.g. I prefer academic tone, I research cancer biology...",
                                     key="new_mem_fact", label_visibility="collapsed")
            if st.form_submit_button("Add Memory", type="primary") and new_fact.strip():
                facts.append(new_fact.strip())
                profile["memory_facts"] = facts
                save_user_data(st.session_state.user_data)
                st.success("✅ Memory saved!"); st.rerun()
        if facts and st.button("🗑️ Clear All Memory", key="clr_mem"):
            profile["memory_facts"] = []
            save_user_data(st.session_state.user_data)
            st.success("All memory cleared."); time.sleep(0.4); st.rerun()
        if st.button("← Back to Dashboard", key="mem_back"):
            st.session_state.current_view = "home"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Analytics
# ══════════════════════════════════════════════════════════════════════════════
def render_analytics():
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown("""
        <div class="settings-heading">📊 Analytics</div>
        <div class="settings-sub">Insights into your research activity</div>
        """, unsafe_allow_html=True)

        chats   = st.session_state.user_data.get("chats",[])
        visible = [c for c in chats if c.get("messages")]
        total_sessions = len(visible)
        total_msgs     = sum(len(c.get("messages",[])) for c in visible)
        total_reports  = sum(1 for c in visible for m in c.get("messages",[])
                             if m.get("type")=="research" and m.get("data",{}).get("report"))
        models_used = {}
        for c in visible:
            m = c.get("model","Unknown")
            models_used[m] = models_used.get(m,0) + 1

        s1,s2,s3,s4 = st.columns(4)
        for col_c, icon, val, label in [
            (s1,"💬",str(total_sessions),"SESSIONS"),
            (s2,"📨",str(total_msgs),"MESSAGES"),
            (s3,"📄",str(total_reports),"REPORTS"),
            (s4,"🤖",str(len(models_used)),"MODELS USED"),
        ]:
            with col_c:
                st.markdown(f"""
                <div class="info-card" style="text-align:center;">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#a78bfa;margin:4px 0;">{val}</div>
                    <div style="font-size:0.68rem;color:rgba(232,232,240,0.35);font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        if models_used:
            st.markdown('<span class="section-label">Model Usage</span>', unsafe_allow_html=True)
            for model, count in sorted(models_used.items(), key=lambda x:-x[1]):
                pct = int(count / max(total_sessions,1) * 100)
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.82rem;color:#c0c0e0;">{html_escape(str(model), quote=True)}</span>
                        <span style="font-size:0.78rem;color:rgba(232,232,240,0.35);font-family:'JetBrains Mono',monospace;">{count} sessions ({pct}%)</span>
                    </div>
                    <div style="background:rgba(139,92,246,0.08);border-radius:4px;height:6px;overflow:hidden;">
                        <div style="width:{pct}%;background:linear-gradient(90deg,#7c3aed,#a78bfa);height:100%;border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if visible:
            st.markdown('<span class="section-label">Recent Sessions</span>', unsafe_allow_html=True)
            for c in sorted(visible, key=lambda x:x.get("timestamp",""), reverse=True)[:10]:
                try:
                    ts = datetime.fromisoformat(c.get("timestamp","")).strftime("%b %d, %Y")
                except Exception:
                    ts = "Unknown"
                mc = len(c.get("messages",[]))
                st.markdown(f"""
                <div class="info-card" style="margin-bottom:6px;">
                    <div class="info-card-row">
                        <span style="font-size:1rem;">💬</span>
                        <div style="flex:1;">
                            <div class="info-card-title">{html_escape(str(c.get('title','Untitled'))[:55], quote=True)}</div>
                            <div class="info-card-sub">{html_escape(ts, quote=True)} · {mc} messages · {html_escape(str(c.get('model','Unknown')), quote=True)}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("← Back to Dashboard", key="ana_back"):
            st.session_state.current_view = "home"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
view = st.session_state.current_view

if view == "welcome":
    render_welcome()
else:
    render_sidebar()
    if   view == "home":      render_home()
    elif view == "chat":      render_chat()
    elif view == "library":   render_local_library()
    elif view == "settings":  render_settings()
    elif view == "memory":    render_memory()
    elif view == "analytics": render_analytics()
    else:                     render_home()

