"""Small JSON storage helpers shared by the UI and compatibility tests."""

import json
import os
import uuid


def _normalized_user_data(data):
    """Return a UI-safe history structure while preserving valid legacy fields."""
    if not isinstance(data, dict):
        return None

    normalized = dict(data)
    profile_defaults = {
        "full_name": "",
        "display_name": "",
        "avatar": "🧑‍💻",
        "work_description": "",
        "org_id": str(uuid.uuid4())[:8].upper(),
        "memory_facts": [],
    }
    profile = data.get("profile")
    profile = dict(profile) if isinstance(profile, dict) else {}
    for key, value in profile_defaults.items():
        profile.setdefault(key, value)
    for key in ("full_name", "display_name", "avatar", "work_description", "org_id"):
        if not isinstance(profile.get(key), str):
            profile[key] = profile_defaults[key]
    if not isinstance(profile.get("memory_facts"), list):
        profile["memory_facts"] = []
    else:
        profile["memory_facts"] = [fact for fact in profile["memory_facts"] if isinstance(fact, str)]
    normalized["profile"] = profile

    settings_defaults = {
        "font": "Inter",
        "language": "English",
        "style": "Balanced",
        "notifications": True,
        "show_thinking": True,
        "auto_title": True,
        "theme": "Dark",
    }
    settings = data.get("settings")
    settings = dict(settings) if isinstance(settings, dict) else {}
    for key, value in settings_defaults.items():
        if key not in settings or not isinstance(settings[key], type(value)):
            settings[key] = value
    normalized["settings"] = settings

    chats = data.get("chats")
    safe_chats = []
    if isinstance(chats, list):
        for raw_chat in chats:
            if not isinstance(raw_chat, dict):
                continue
            chat = dict(raw_chat)
            if not isinstance(chat.get("id"), str) or not chat["id"]:
                chat["id"] = str(uuid.uuid4())
            if not isinstance(chat.get("title"), str):
                chat["title"] = "New Research"
            messages = chat.get("messages")
            safe_messages = []
            if isinstance(messages, list):
                for raw_message in messages:
                    if not isinstance(raw_message, dict):
                        continue
                    role = raw_message.get("role")
                    if not isinstance(role, str) or role not in {"user", "assistant"}:
                        continue
                    message = dict(raw_message)
                    if message["role"] == "user":
                        if not isinstance(message.get("content"), str):
                            message["content"] = ""
                        files = message.get("files")
                        message["files"] = [item for item in files if isinstance(item, dict)] if isinstance(files, list) else []
                    elif message.get("type") == "research" and not isinstance(message.get("data"), dict):
                        message["data"] = {}
                    if message.get("role") == "assistant" and message.get("type") == "research":
                        research_data = dict(message.get("data") or {})
                        for key in ("search", "reader", "report", "critic"):
                            if not isinstance(research_data.get(key), str):
                                research_data[key] = ""
                        sources = research_data.get("sources")
                        research_data["sources"] = [item for item in sources if isinstance(item, dict)] if isinstance(sources, list) else []
                        message["data"] = research_data
                    safe_messages.append(message)
            chat["messages"] = safe_messages
            safe_chats.append(chat)
    normalized["chats"] = safe_chats

    projects = data.get("projects")
    safe_projects = []
    if isinstance(projects, list):
        for raw_project in projects:
            if not isinstance(raw_project, dict):
                continue
            project = dict(raw_project)
            if not isinstance(project.get("id"), str) or not project["id"]:
                project["id"] = str(uuid.uuid4())
            if not isinstance(project.get("name"), str):
                project["name"] = ""
            safe_projects.append(project)
    normalized["projects"] = safe_projects
    return normalized


def load_user_data(data_file: str):
    if os.path.exists(data_file):
        try:
            with open(data_file, "r", encoding="utf-8") as handle:
                return _normalized_user_data(json.load(handle))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
    return None


def save_user_data(data, data_file: str) -> None:
    try:
        with open(data_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
    except OSError:
        pass
