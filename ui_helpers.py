"""Pure helpers for small, HTML-backed Streamlit UI fragments."""

from __future__ import annotations

from html import escape
import re
from textwrap import dedent
from typing import MutableMapping


def normalize_theme(theme: str | None) -> str:
    """Return one of the two supported local UI themes."""
    return "Light" if str(theme or "").strip().lower() == "light" else "Dark"


def set_user_theme(user_data: dict, theme: str) -> str:
    """Persist a normalized visual preference without touching chat history."""
    selected = normalize_theme(theme)
    settings = user_data.setdefault("settings", {})
    if not isinstance(settings, dict):
        settings = {}
        user_data["settings"] = settings
    settings["theme"] = selected
    return selected


def start_new_chat_state(state: MutableMapping) -> None:
    """Return the UI to a clean composer while retaining all saved chats."""
    state["current_view"] = "home"
    state["current_chat_id"] = None
    state["pipeline_running"] = False
    state["pipeline_topic"] = ""


def model_status_label(provider_state: str, selected_model: str, local_model: str) -> str:
    """Describe observed application state without probing providers or implying reachability."""
    labels = {
        "not_checked": "Provider status not checked this session",
        "api_success": "OpenRouter generation succeeded",
        "api_unconfigured": "OpenRouter configuration missing; local fallback used",
        "api_fallback": "Remote research unavailable; local fallback used",
        "local_file": "Project file answered locally; providers not contacted",
        "generation_error": "Generation failed; see the message for safe details",
    }
    api_status = labels.get(str(provider_state), labels["not_checked"])
    return f"{api_status} · Local fallback: {local_model} (connection not checked) · Selected: {selected_model}"


def safe_image_data_uri(value: str) -> str | None:
    """Allow only bounded-format raster image previews in HTML-backed messages."""
    candidate = str(value or "")
    if len(candidate) > 8_000_000:
        return None
    if re.fullmatch(r"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/]*={0,2}", candidate):
        return candidate
    return None


def build_user_message_html(content: str, files: list[dict] | None = None, timestamp: str = "") -> str:
    """Render escaped user content and only allow validated raster previews."""
    attachments = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        preview = safe_image_data_uri(item.get("preview", ""))
        if preview:
            attachments.append(
                '<div class="user-image-preview"><img src="'
                + preview
                + '" alt="User attached image"></div>'
            )
        else:
            file_type = str(item.get("type", ""))
            icon = "🖼️" if "image" in file_type else "📄"
            name = escape(str(item.get("name", "file"))[:80], quote=True)
            attachments.append(f'<span class="attach-chip">{icon} {name}</span>')
    files_html = '<div class="message-attachments">' + "".join(attachments) + "</div>" if attachments else ""
    return dedent(f"""
    <div class="msg-user-wrap">
        <div>
            <div class="msg-user">{escape(str(content or ""), quote=True)}{files_html}</div>
            <div class="msg-ts">{escape(str(timestamp or ""), quote=True)}</div>
        </div>
    </div>
    """).strip()


def build_chat_topbar_html(
    title: str,
    selected_model: str,
    *,
    last_response_model: str | None = None,
    incognito: bool = False,
) -> str:
    """Show configured API model separately from the model that answered last."""
    safe_title = escape(str(title), quote=True)
    badge_text = f"Configured API: {selected_model}"
    if last_response_model:
        badge_text += f" · Last answer: {last_response_model}"
    safe_model = escape(badge_text, quote=True)
    incognito_badge = (
        '<span class="incognito-badge" style="margin-left:10px;">🛡️ Incognito</span>'
        if incognito else ""
    )
    # Keep the HTML contiguous: when the optional incognito badge is empty, a
    # blank line can terminate Streamlit's HTML block and expose the next span
    # as literal markup.
    return (
        '<div class="chat-topbar">'
        f'<span class="chat-topbar-title">{safe_title}</span>'
        f'{incognito_badge}'
        f'<span class="topbar-badge">{safe_model}</span>'
        '</div>'
    )
