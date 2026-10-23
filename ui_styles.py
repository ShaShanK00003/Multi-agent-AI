"""Theme tokens and responsive overrides for the Streamlit application shell.

The base stylesheet remains in app.py to preserve its established visual design.
These overrides centralize the theme-dependent palette and accessible shell states.
Streamlit currently exposes several useful regions through data-testid attributes;
those selectors are a practical integration point but may need review after upgrades.
"""

from __future__ import annotations

from html import escape

from ui_helpers import normalize_theme


THEME_TOKENS = {
    "Dark": {
        "bg": "#111214", "sidebar": "#17181b", "surface": "#1d1e21", "surface_alt": "#27282c",
        "text": "#f3f4f6", "muted": "#b4b6bd", "border": "#37393f", "accent": "#a6b5ff",
        "accent_strong": "#5268d8", "focus": "#a6b5ff", "code": "#17181b", "success": "#16794a",
    },
    "Light": {
        "bg": "#f7f7f8", "sidebar": "#fcfcfd", "surface": "#ffffff", "surface_alt": "#f1f2f4",
        "text": "#202124", "muted": "#5f636b", "border": "#dedfe3", "accent": "#3949ab",
        "accent_strong": "#4358c7", "focus": "#3949ab", "code": "#f2f3f5", "success": "#17633e",
    },
}

FONT_STACKS = {
    "Inter": 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "System Default": '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "Mono": 'ui-monospace, SFMono-Regular, Consolas, monospace',
    "Serif": 'Georgia, "Times New Roman", serif',
}


def build_theme_css(theme: str, font: str = "Inter") -> str:
    """Build a safe CSS palette and shell overrides for Dark or Light mode."""
    selected = normalize_theme(theme)
    tokens = THEME_TOKENS[selected]
    font_stack = FONT_STACKS.get(str(font), FONT_STACKS["Inter"])
    variables = "\n".join(f"  --rm-{escape(name)}: {escape(value)};" for name, value in tokens.items())
    return f"""
<style>
:root {{
{variables}
  color-scheme: {'dark' if selected == 'Dark' else 'light'};
  --rm-radius: 14px;
  --rm-shadow: 0 12px 36px rgba(20, 22, 32, .12);
}}
html, body, [class*="css"] {{ font-family: {font_stack} !important; }}
body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--rm-bg) !important; color: var(--rm-text) !important; }}
.block-container {{
  width: 100% !important; max-width: 1180px !important; margin: 0 auto !important;
  padding: 24px 32px 72px !important; color: var(--rm-text) !important;
}}
.st-key-home-workspace {{ width: 100%; max-width: 860px; margin: clamp(28px, 9vh, 112px) auto 0; padding-bottom: 3rem; }}
.home-hero {{ margin: 0 0 32px; }}
.home-kicker {{ color: var(--rm-accent); font-size: .7rem; font-weight: 650; letter-spacing: .14em; margin-bottom: 12px; }}
.home-greeting {{ color: var(--rm-text) !important; text-align: left !important; font-size: clamp(2rem, 4.2vw, 3rem); font-weight: 650; line-height: 1.12; letter-spacing: -.04em; margin: 0; }}
.home-subtitle {{ color: var(--rm-muted) !important; font-size: 1rem; margin: 10px 0 0; }}
.home-suggestions-label {{ color: var(--rm-muted); font-size: .78rem; margin: 0 0 10px; }}
.st-key-home-workspace [data-testid="stButton"] button {{
  height: auto; min-height: 56px; white-space: normal; text-align: left; justify-content: flex-start;
  line-height: 1.35; padding: 12px 14px !important; border-radius: 12px !important;
}}
.st-key-home-workspace [data-testid="stForm"] {{
  background: var(--rm-surface) !important; border: 1px solid var(--rm-border) !important;
  border-radius: 18px !important; padding: 14px 16px 12px !important; margin-top: 18px;
  box-shadow: var(--rm-shadow);
}}
.st-key-home-workspace [data-testid="stTextArea"] textarea {{
  border: 1px solid var(--rm-border) !important; background: var(--rm-surface) !important;
  color: var(--rm-text) !important; box-shadow: none !important;
  padding: 10px 4px !important; min-height: 76px;
}}
html body .stApp .st-key-home-workspace textarea {{
  background-color: var(--rm-surface) !important; color: var(--rm-text) !important;
  border-color: var(--rm-border) !important;
}}
.st-key-home-workspace [data-testid="stFileUploaderDropzone"] {{ min-height: 50px; }}
.st-key-home-workspace [data-testid="stCaptionContainer"] {{ line-height: 1.5; }}
[data-testid="stHeader"] {{ background: color-mix(in srgb, var(--rm-bg) 92%, transparent) !important; }}
/* Keep the existing chrome hidden until Streamlit needs its sidebar reopen button. */
body:has([data-testid="stExpandSidebarButton"]) [data-testid="stHeader"] {{ display: flex !important; visibility: visible !important; }}
[data-testid="stSidebar"] {{ background: var(--rm-sidebar) !important; border-right: 1px solid var(--rm-border) !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{ color: var(--rm-text); }}
[data-testid="stSidebar"] button {{ color: var(--rm-text) !important; }}
[data-testid="stSidebar"] button[kind="primary"] {{ background: var(--rm-accent-strong) !important; color: #fff !important; border-color: var(--rm-accent-strong) !important; }}
[data-testid="stSidebar"] button:disabled {{ color: var(--rm-muted) !important; opacity: .65; }}
[data-testid="stSidebar"] hr {{ border-color: var(--rm-border) !important; }}
button, [role="button"] {{ min-height: 40px; }}
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="combobox"]:focus-visible,
[tabindex]:focus-visible {{ outline: 3px solid var(--rm-focus) !important; outline-offset: 2px !important; }}
.block-container button[kind="secondary"] {{ background: var(--rm-surface-alt) !important; color: var(--rm-text) !important; border-color: var(--rm-border) !important; }}
html body .stApp input:not([type="file"]),
html body .stApp textarea,
html body .stApp [data-baseweb="select"],
html body .stApp [data-baseweb="select"] > div {{
  background: var(--rm-surface) !important; color: var(--rm-text) !important;
  border-color: var(--rm-border) !important;
}}
html body .stApp [data-baseweb="select"] span,
html body .stApp [data-baseweb="select"] [role="combobox"] {{ color: var(--rm-text) !important; }}
html body .stApp input::placeholder,
html body .stApp textarea::placeholder {{ color: var(--rm-muted) !important; opacity: 1 !important; }}
.stApp label {{ color: var(--rm-muted) !important; }}
.stTextInput label, .stTextArea label, .stSelectbox label, .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--rm-muted) !important; }}
[data-baseweb="menu"], [role="listbox"] {{ background: var(--rm-surface) !important; border: 1px solid var(--rm-border) !important; }}
[role="option"] {{ color: var(--rm-text) !important; }}
[role="option"]:hover {{ background: var(--rm-surface-alt) !important; }}
.stApp div[data-testid="stFileUploader"] label,
.stApp [data-testid="stFileUploaderDropzone"],
.stApp [data-testid="stFileUploaderDropzone"] * {{ color: var(--rm-text) !important; }}
.stApp [data-testid="stFileUploaderDropzone"] {{ background: var(--rm-surface-alt) !important; border-color: var(--rm-border) !important; }}
.stApp [data-testid="stFileUploaderDropzone"] button {{
  background: var(--rm-surface) !important; color: var(--rm-text) !important; border-color: var(--rm-border) !important;
}}
[data-testid="stPopover"] > div > button {{ background: var(--rm-surface) !important; color: var(--rm-text) !important; border-color: var(--rm-border) !important; }}
[data-testid="stExpander"] {{ background: var(--rm-surface) !important; border-color: var(--rm-border) !important; }}
[data-testid="stChatInput"] {{ background: var(--rm-bg) !important; border-color: var(--rm-border) !important; }}
[data-testid="stStatus"], .stAlert, details, .report-card, .info-card, .model-card,
.memory-card, .typing-bubble, .chat-topbar {{ background-color: var(--rm-surface) !important; border-color: var(--rm-border) !important; color: var(--rm-text) !important; }}
.report-card, .info-card, .model-card, .memory-card {{ border-radius: var(--rm-radius) !important; box-shadow: var(--rm-shadow); }}
.home-greeting, .chat-topbar-title, .msg-ai-body, .msg-ai-body p, .msg-ai-body li,
.msg-ai-name, .settings-heading, .info-card-title, .memory-fact, .model-name,
.welcome-title, .welcome-feat-title, .msg-ai-body h1, .msg-ai-body h2, .msg-ai-body strong {{ color: var(--rm-text) !important; }}
.welcome-title-accent, .msg-ai-body h3 {{ color: var(--rm-accent) !important; }}
.home-metric-value {{ font-size: 1.4rem; font-weight: 700; color: var(--rm-accent) !important; }}
.home-metric-label {{ font-size: .72rem; color: var(--rm-muted) !important; margin-top: 2px; font-family: ui-monospace, Consolas, monospace; letter-spacing: .05em; }}
.msg-user {{ white-space: pre-wrap; overflow-wrap: anywhere; background: var(--rm-accent-strong) !important; color: #fff !important; border-color: var(--rm-accent-strong) !important; }}
.message-attachments {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
.user-image-preview {{ border: 1px solid var(--rm-border); border-radius: 8px; overflow: hidden; }}
.user-image-preview img {{ display: block; max-width: 200px; max-height: 140px; object-fit: contain; }}
.msg-ai-body code {{ color: var(--rm-accent) !important; background: var(--rm-surface-alt) !important; border-color: var(--rm-border) !important; }}
.msg-ai-body h1 {{ border-color: var(--rm-border) !important; }}
.home-subtitle, .settings-sub, .model-desc, .model-provider, .info-card-sub,
.msg-ts, .app-footer, .welcome-feat-desc, .sb-brand-version, .model-ctx, .input-hint {{ color: var(--rm-muted) !important; opacity: .9; }}
.home-chip {{ color: var(--rm-text) !important; background: var(--rm-surface-alt) !important; border-color: var(--rm-border) !important; }}
.section-label {{ color: var(--rm-accent) !important; border-color: var(--rm-border) !important; }}
.sb-brand-text, .sb-section-label, .attach-chip, .topbar-badge {{ color: var(--rm-text) !important; }}
.model-pill {{ color: var(--rm-accent) !important; background: var(--rm-surface-alt) !important; border-color: var(--rm-border) !important; }}
.welcome-sub {{ color: var(--rm-muted) !important; }}
.model-card.selected {{ border-color: var(--rm-accent) !important; background: var(--rm-surface-alt) !important; }}
.msg-ai-body blockquote {{ color: var(--rm-muted) !important; border-color: var(--rm-accent) !important; }}
.msg-ai-body td {{ color: var(--rm-text) !important; border-color: var(--rm-border) !important; }}
.msg-ai-body th {{ color: var(--rm-accent) !important; background: var(--rm-surface-alt) !important; border-color: var(--rm-border) !important; }}
.msg-ai-body pre, .stCode, [data-testid="stCode"] {{ background: var(--rm-code) !important; color: var(--rm-text) !important; max-width: 100%; overflow-x: auto !important; }}
.msg-ai-body table {{ display: block; max-width: 100%; overflow-x: auto; }}
.report-badge {{ color: var(--rm-accent) !important; }}
.pipeline-step.active {{ color: var(--rm-accent) !important; }}
[data-testid="stSidebarCollapsedControl"] {{ background: var(--rm-surface) !important; border-color: var(--rm-border) !important; }}
/* ChatGPT/Claude-inspired application chrome using native Streamlit controls. */
.stApp {{ letter-spacing: -0.006em; }}
[data-testid="stSidebar"] {{ padding-top: 10px; }}
[data-testid="stSidebar"] [data-testid="stButton"] button {{
  min-height: 40px; border-radius: 10px !important; text-align: left;
  transition: background-color .15s ease, border-color .15s ease;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
  background: var(--rm-surface-alt) !important; border-color: var(--rm-border) !important;
}}
.sb-brand {{ padding: 8px 8px 18px; }}
.sb-brand-text {{ font-size: 1.05rem !important; letter-spacing: -.025em; }}
.sb-brand-version {{ font-size: .68rem; }}
.st-key-home-workspace {{ max-width: 820px; margin-top: clamp(52px, 12vh, 132px); }}
.home-hero {{ text-align: center; margin-bottom: 26px; }}
.home-kicker {{ font-size: .68rem; letter-spacing: .16em; }}
.home-greeting {{ text-align: center !important; font-weight: 560; letter-spacing: -.045em; }}
.home-subtitle {{ margin-top: 12px; }}
.st-key-home-workspace [data-testid="stForm"] {{
  border-radius: 22px !important; padding: 16px 18px 14px !important;
  box-shadow: 0 8px 28px rgba(0,0,0,.08);
}}
.st-key-home-workspace [data-testid="stTextArea"] textarea {{
  min-height: 94px; border: 0 !important; border-radius: 12px !important;
  font-size: 1rem; line-height: 1.55; resize: vertical;
}}
.st-key-home-workspace [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stChatInput"] button {{ border-radius: 12px !important; font-weight: 600; }}
.stApp [data-testid="stChatInput"] {{ border-radius: 20px !important; box-shadow: 0 6px 24px rgba(0,0,0,.08); }}
.chat-topbar {{ min-height: 54px; border-bottom: 1px solid var(--rm-border) !important; }}
.topbar-badge {{ border-radius: 999px !important; padding: 5px 10px !important; font-size: .72rem !important; }}
.msg-ai-body {{ max-width: 820px; line-height: 1.72; font-size: .98rem; }}
.msg-ai-body p {{ line-height: 1.72; margin: 0 0 1em; }}
.msg-user {{ border-radius: 18px !important; background: var(--rm-surface-alt) !important; color: var(--rm-text) !important; }}
.report-card {{ border-radius: 18px !important; padding: 22px !important; }}
.stAlert {{ border-radius: 12px !important; }}
button, input, textarea, [role="combobox"] {{ transition: border-color .15s ease, box-shadow .15s ease; }}
::selection {{ background: color-mix(in srgb, var(--rm-accent) 35%, transparent); }}
::-webkit-scrollbar-thumb {{ background: color-mix(in srgb, var(--rm-accent) 45%, transparent); border-radius: 8px; }}
@media (max-width: 760px) {{
  .block-container {{ padding: 16px 14px 88px !important; }}
  .st-key-home-workspace {{ margin-top: 5vh; }}
  .home-greeting {{ font-size: clamp(1.8rem, 8vw, 2.5rem) !important; }}
  .st-key-home-workspace [data-testid="stForm"] {{ padding: 10px !important; }}
  .st-key-home-workspace [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .chat-topbar {{ padding: 10px 12px !important; gap: 8px !important; }}
  .chat-topbar-title {{ min-width: 0; }}
  .msg-user-wrap>div {{ max-width: 92% !important; }}
  .report-card {{ padding: 16px !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: 8px 10px !important; }}
  .stDownloadButton button {{ white-space: normal !important; }}
}}
@media (max-width: 900px) {{
  /* Four weighted controls become too narrow beside the open sidebar. */
  .st-key-home-workspace [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
  .st-key-home-workspace [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    flex: 1 1 170px !important; min-width: 170px !important;
  }}
  .st-key-home-workspace [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
    white-space: normal !important; overflow-wrap: anywhere;
  }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ scroll-behavior: auto !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }}
}}
</style>
"""
