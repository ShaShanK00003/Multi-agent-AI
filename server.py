"""
ResearchMind Flask Server — Production-ready backend for the new ChatGPT/Claude-style UI.
Replaces Streamlit. All existing Python logic (agents, pipeline, Ollama, ChromaDB) is reused.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import markdown2
from dotenv import load_dotenv, set_key
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "user_data.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.secret_key = os.urandom(24)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Import existing backend modules ───────────────────────────────────────────
from history_store import load_user_data as _load_user_data, save_user_data as _save_user_data
from local_llm import DEFAULT_OLLAMA_MODEL, LOCAL_MODEL_OPTIONS, get_configured_local_model
from model_config import MODEL_CONFIGS

# ── Constants ─────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "GPT-4o":            {"provider": "OpenAI",    "desc": "Most capable GPT model",       "badge": "Flagship", "ctx": "128K"},
    "GPT-4o Mini":       {"provider": "OpenAI",    "desc": "Fast & cost-effective",         "badge": "Fast",     "ctx": "128K"},
    "Claude 3.5 Sonnet": {"provider": "Anthropic", "desc": "Intelligent & nuanced writing", "badge": "Best",     "ctx": "200K"},
    "Claude 3 Haiku":    {"provider": "Anthropic", "desc": "Ultra-fast responses",          "badge": "Fast",     "ctx": "200K"},
    "Gemini 1.5 Pro":    {"provider": "Google",    "desc": "Multimodal & long context",     "badge": "Pro",      "ctx": "1M"},
    "Llama 3.1 70B":     {"provider": "Meta",      "desc": "Powerful open-source model",   "badge": "Open",     "ctx": "128K"},
    "Mixtral 8x22B":     {"provider": "Mistral",   "desc": "High quality MoE model",       "badge": "MoE",      "ctx": "65K"},
    "DeepSeek R1":       {"provider": "DeepSeek",  "desc": "Advanced reasoning model",      "badge": "Reasoning","ctx": "64K"},
}
for _label, _info in AVAILABLE_MODELS.items():
    _info["api"] = MODEL_CONFIGS.get(_label, "openai/gpt-4o-mini")

SUGGESTIONS = [
    {"emoji": "🔬", "text": "Quantum computing breakthroughs 2025"},
    {"emoji": "🧬", "text": "CRISPR gene editing clinical trials progress"},
    {"emoji": "⚡", "text": "Nuclear fusion energy — commercial timeline"},
    {"emoji": "🤖", "text": "Multi-agent AI systems architecture"},
    {"emoji": "🌍", "text": "Carbon capture technology effectiveness"},
    {"emoji": "💊", "text": "mRNA technology beyond COVID vaccines"},
    {"emoji": "🚀", "text": "SpaceX Starship missions and Mars plans"},
    {"emoji": "🧠", "text": "Neuroplasticity and cognitive enhancement"},
]


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_user_data():
    return _load_user_data(str(DATA_FILE))


def save_user_data(data):
    _save_user_data(data, str(DATA_FILE))


def get_default_data():
    return {
        "profile": {
            "full_name": "",
            "display_name": "",
            "avatar": "🧑\u200d💻",
            "work_description": "",
            "org_id": str(uuid.uuid4())[:8].upper(),
            "memory_facts": [],
        },
        "settings": {
            "font": "Inter",
            "language": "English",
            "style": "Balanced",
            "notifications": True,
            "show_thinking": True,
            "local_model": DEFAULT_OLLAMA_MODEL,
            "theme": "dark",
            "auto_title": True,
        },
        "chats": [],
        "projects": [],
    }


def get_or_create_data():
    data = load_user_data()
    if not isinstance(data, dict):
        data = get_default_data()
        save_user_data(data)
    return data


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/user", methods=["GET"])
def get_user():
    data = get_or_create_data()
    return jsonify(data)


@app.route("/api/user/profile", methods=["POST"])
def update_profile():
    data = get_or_create_data()
    body = request.get_json(force=True) or {}
    for k in ("full_name", "display_name", "avatar", "work_description"):
        if k in body:
            data["profile"][k] = body[k]
    save_user_data(data)
    return jsonify({"ok": True, "profile": data["profile"]})


@app.route("/api/user/settings", methods=["POST"])
def update_settings():
    data = get_or_create_data()
    body = request.get_json(force=True) or {}
    data.setdefault("settings", {}).update(body)
    save_user_data(data)
    return jsonify({"ok": True, "settings": data["settings"]})


@app.route("/api/user/memory", methods=["GET", "POST", "DELETE"])
def manage_memory():
    data = get_or_create_data()
    facts = data["profile"].setdefault("memory_facts", [])
    if request.method == "GET":
        return jsonify(facts)
    if request.method == "POST":
        body = request.get_json(force=True) or {}
        fact = (body.get("fact") or "").strip()
        if fact:
            facts.append(fact)
            save_user_data(data)
        return jsonify(facts)
    if request.method == "DELETE":
        body = request.get_json(force=True) or {}
        idx = body.get("index")
        if idx is not None and 0 <= int(idx) < len(facts):
            facts.pop(int(idx))
            save_user_data(data)
        return jsonify(facts)


@app.route("/api/chats", methods=["GET"])
def get_chats():
    data = get_or_create_data()
    chats = [c for c in data.get("chats", []) if not c.get("incognito") and c.get("messages")]
    chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(chats)


@app.route("/api/chats", methods=["POST"])
def create_chat_api():
    data = get_or_create_data()
    body = request.get_json(force=True) or {}
    chat = {
        "id": str(uuid.uuid4()),
        "title": body.get("title", "New Research"),
        "messages": [],
        "timestamp": datetime.now().isoformat(),
        "incognito": body.get("incognito", False),
        "pinned": False,
        "model": body.get("model", "Claude 3.5 Sonnet"),
        "branch_of": None,
    }
    if not chat["incognito"]:
        data.setdefault("chats", []).append(chat)
        save_user_data(data)
    return jsonify(chat)


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    data = get_or_create_data()
    for c in data.get("chats", []):
        if c["id"] == chat_id:
            return jsonify(c)
    return jsonify({"error": "Not found"}), 404


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    data = get_or_create_data()
    data["chats"] = [c for c in data.get("chats", []) if c["id"] != chat_id]
    save_user_data(data)
    return jsonify({"ok": True})


@app.route("/api/chats/clear", methods=["POST"])
def clear_chats():
    data = get_or_create_data()
    data["chats"] = []
    save_user_data(data)
    return jsonify({"ok": True})


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify(AVAILABLE_MODELS)


@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    return jsonify(SUGGESTIONS)


@app.route("/api/apikeys", methods=["POST"])
def save_api_keys():
    body = request.get_json(force=True) or {}
    env_path = str(BASE_DIR / ".env")
    try:
        or_key = body.get("openrouter_key", "").strip()
        tav_key = body.get("tavily_key", "").strip()
        if or_key:
            set_key(env_path, "OPENROUTER_API_KEY", or_key, quote_mode="always")
            os.environ["OPENROUTER_API_KEY"] = or_key
        if tav_key:
            set_key(env_path, "TAVILY_API_KEY", tav_key, quote_mode="always")
            os.environ["TAVILY_API_KEY"] = tav_key
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    data = get_or_create_data()
    chats = [c for c in data.get("chats", []) if c.get("messages")]
    total_sessions = len(chats)
    total_msgs = sum(len(c.get("messages", [])) for c in chats)
    total_reports = sum(
        1 for c in chats
        for m in c.get("messages", [])
        if m.get("type") == "research" and m.get("data", {}).get("report")
    )
    models_used = {}
    for c in chats:
        m = c.get("model", "Unknown")
        models_used[m] = models_used.get(m, 0) + 1
    recent = sorted(chats, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]
    recent_summary = [
        {
            "title": c.get("title", "Untitled")[:60],
            "timestamp": c.get("timestamp", ""),
            "model": c.get("model", "Unknown"),
            "message_count": len(c.get("messages", [])),
        }
        for c in recent
    ]
    return jsonify({
        "total_sessions": total_sessions,
        "total_messages": total_msgs,
        "total_reports": total_reports,
        "models_used": models_used,
        "recent_sessions": recent_summary,
    })


@app.route("/api/research", methods=["POST"])
def run_research_endpoint():
    """Main research endpoint with SSE streaming."""
    body = request.get_json(force=True) or {}
    topic = (body.get("topic") or "").strip()
    model_name = body.get("model", "Claude 3.5 Sonnet")
    chat_id = body.get("chat_id")
    history = body.get("history", [])

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    model_api = AVAILABLE_MODELS.get(model_name, {}).get("api", "openai/gpt-4o-mini")

    def generate():
        data = get_or_create_data()
        local_model = get_configured_local_model(data.get("settings"))

        yield _sse("status", {"step": "starting", "message": "🔬 Initializing research pipeline..."})
        time.sleep(0.1)

        try:
            from chat_service import run_research_with_local_fallback
            from research_pipeline import sources_for_history

            yield _sse("status", {"step": "searching", "message": "🔍 Searching web and local knowledge base..."})

            result = run_research_with_local_fallback(
                topic, model_api, history=history, local_model=local_model,
            )

            generation_mode = result.get("generation_mode", "api_research")
            fallback = result.get("fallback_notice")

            if generation_mode == "local":
                yield _sse("status", {"step": "local", "message": "🖥️ Answering from local knowledge..."})
            else:
                yield _sse("status", {"step": "writing", "message": "✍️ Writing detailed research report..."})
                time.sleep(0.1)
                yield _sse("status", {"step": "reviewing", "message": "🧐 Running AI peer review..."})

            report_html = markdown2.markdown(
                result.get("report", ""),
                extras=["tables", "fenced-code-blocks", "strike", "task_list"],
            )
            critic_html = markdown2.markdown(
                result.get("feedback", "") or "",
                extras=["tables", "fenced-code-blocks"],
            )

            assistant_msg = {
                "role": "assistant",
                "type": result.get("message_type", "research"),
                "data": {
                    "report": result.get("report", ""),
                    "report_html": report_html,
                    "critic": result.get("feedback", ""),
                    "critic_html": critic_html,
                    "sources": sources_for_history(result.get("sources", [])),
                    "search_error": result.get("search_error"),
                    "local_retrieval_error": result.get("local_retrieval_error"),
                    "citation_validation": result.get("citation_validation"),
                    "generation_mode": generation_mode,
                    "fallback_notice": fallback,
                    "topic_truncated": result.get("topic_truncated", False),
                },
                "timestamp": datetime.now().isoformat(),
                "model": result.get("model_label", model_name),
            }

            if chat_id:
                data = get_or_create_data()
                for c in data.get("chats", []):
                    if c["id"] == chat_id:
                        c["messages"].append(assistant_msg)
                        if c.get("title") in ("New Research", ""):
                            c["title"] = topic[:50]
                        save_user_data(data)
                        break

            yield _sse("done", {"message": assistant_msg, "generation_mode": generation_mode})

        except Exception as exc:
            logger.error("Research error: %s\n%s", type(exc).__name__, traceback.format_exc())
            err_msg = str(exc) if len(str(exc)) < 400 else "An internal error occurred. Please retry."
            yield _sse("error", {"message": err_msg})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/chats/<chat_id>/message", methods=["POST"])
def add_user_message(chat_id):
    data = get_or_create_data()
    body = request.get_json(force=True) or {}
    msg = {
        "role": "user",
        "content": body.get("content", ""),
        "timestamp": datetime.now().isoformat(),
        "files": body.get("files", []),
    }
    for c in data.get("chats", []):
        if c["id"] == chat_id:
            c["messages"].append(msg)
            if c.get("title") in ("New Research", ""):
                c["title"] = msg["content"][:50]
            save_user_data(data)
            return jsonify({"ok": True, "message": msg})
    return jsonify({"error": "Chat not found"}), 404


@app.route("/api/library/documents", methods=["GET"])
def list_library_documents():
    try:
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        docs = kb.list_documents()
        return jsonify(docs)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/library/documents", methods=["POST"])
def index_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    try:
        from document_library import ingest_uploaded_document
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        result = ingest_uploaded_document(kb, f.filename, f.read())
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/library/documents/<doc_id>", methods=["DELETE"])
def delete_library_document(doc_id):
    try:
        from document_library import delete_local_document
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        result = delete_local_document(kb, doc_id, confirmed=True)
        return jsonify(result or {"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/library/search", methods=["POST"])
def search_library():
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify([])
    try:
        from document_library import retrieve_local_documents
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        records = retrieve_local_documents(kb, query, top_k=5)
        return jsonify(records)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ResearchMind AI Research Platform")
    print("  Production Flask Server")
    print("  Open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
