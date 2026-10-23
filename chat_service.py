"""Online research with a transparent local Ollama fallback."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from agents import ModelConfigurationError, validate_model_configuration
from local_llm import (
    DEFAULT_OLLAMA_MODEL,
    LocalModelError,
    generate_local_response,
    resolve_local_model,
)
from research_pipeline import (
    MAX_LOCAL_RESULTS,
    MAX_LOCAL_TEXT_CHARS,
    MAX_TOPIC_CHARS,
    ResearchPipelineError,
    _validate_citations,
    _get_default_knowledge_base,
    run_research,
    sources_for_history,
)
from project_files import (
    MAX_PROJECT_FILE_CONTEXT_CHARS,
    ProjectFileError,
    read_project_file,
    requested_project_paths,
)


class ChatGenerationError(RuntimeError):
    """Safe user-facing error when neither remote nor local generation succeeds."""


def _local_source_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for index, record in enumerate(records, 1):
        text = str(record.get("text") or "")[:MAX_LOCAL_TEXT_CHARS]
        if not text:
            continue
        chunk_id = str(record.get("chunk_id") or "")
        document_id = str(record.get("document_id") or "")
        filename = str(record.get("source") or "Unknown local document")
        identity = chunk_id or document_id or f"{filename}:{text}"
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            identity = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        sources.append({
            "id": f"local-{index}",
            "source_id": f"L{identity}",
            "source_type": "local_document",
            "title": filename,
            "filename": filename,
            "url": None,
            "page_number": record.get("page"),
            "document_id": document_id or None,
            "chunk_id": chunk_id or None,
            "similarity": record.get("similarity"),
            "distance": record.get("distance"),
            "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
            "text": text,
            "retrieved": True,
        })
    return sources


def _local_answer(
    topic: str,
    fallback_notice: str,
    *,
    history: Iterable[dict[str, Any]] | None,
    retrieve_local: bool = True,
    local_model: str | None = None,
) -> dict[str, Any]:
    try:
        selected_local_model = resolve_local_model(local_model)
    except LocalModelError as exc:
        raise ChatGenerationError(str(exc)) from None
    retrieval_error = None
    topic_truncated = len(topic) > MAX_TOPIC_CHARS
    records = []
    if retrieve_local:
        try:
            knowledge_base = _get_default_knowledge_base()
            records = knowledge_base.retrieve(topic[:MAX_TOPIC_CHARS], top_k=MAX_LOCAL_RESULTS) if knowledge_base else []
            records = [record for record in records if isinstance(record, dict) and str(record.get("text") or "").strip()]
        except Exception as exc:
            records = []
            retrieval_error = f"Local document retrieval unavailable ({type(exc).__name__})."

    file_read_errors: list[str] = []
    for relative_path in requested_project_paths(topic):
        try:
            file_text = read_project_file(relative_path)
        except ProjectFileError as exc:
            file_read_errors.append(f"{relative_path}: {exc}")
            continue
        if not file_text.strip():
            file_read_errors.append(f"{relative_path}: the file is empty.")
            continue
        if len(file_text) > MAX_PROJECT_FILE_CONTEXT_CHARS:
            file_read_errors.append(
                f"{relative_path}: the file exceeds the {MAX_PROJECT_FILE_CONTEXT_CHARS}-character model context limit."
            )
            continue
        chunk_id = hashlib.sha256((relative_path + "\0" + file_text).encode("utf-8")).hexdigest()
        records.insert(0, {
            "text": file_text,
            "source": relative_path,
            "page": None,
            "document_id": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
            "chunk_id": chunk_id,
            "metadata": {"source_type": "project_file", "path": relative_path},
        })

    sources = _local_source_records(records)
    for record, source in zip(records, sources):
        record["source_id"] = source["source_id"]
        record["text"] = source["text"]
    try:
        report = generate_local_response(
            topic,
            records,
            history,
            context_note=("Requested project file could not be read. Do not guess its contents or versions. "
                          + "; ".join(file_read_errors)) if file_read_errors else None,
            model=selected_local_model,
        )
    except LocalModelError as exc:
        raise ChatGenerationError(
            f"{fallback_notice} Local Ollama ({selected_local_model}) could not generate an answer: {exc}"
        ) from None

    if sources:
        report, citation_validation = _validate_citations(report, sources)
    else:
        citation_validation = {
            "valid": True,
            "cited_source_ids": [],
            "unknown_source_ids": [],
            "validation_type": "not_applicable_general_chat",
        }
    return {
        "search_results": "Web research was skipped; this answer was generated locally.",
        "scraped_content": "No webpages were fetched in local mode.",
        "report": report,
        "feedback": "Peer review was not run in local mode.",
        "sources": sources,
        "search_error": "Skipped in local mode.",
        "local_retrieval_error": retrieval_error,
        "citation_validation": citation_validation,
        "topic_truncated": topic_truncated,
        "generation_mode": "local",
        "message_type": "local_chat",
        "model_label": f"{selected_local_model} (local)",
        "fallback_notice": fallback_notice,
    }


def run_research_with_local_fallback(
    topic: str,
    model_name: str | None = None,
    *,
    history: Iterable[dict[str, Any]] | None = None,
    local_model: str | None = None,
) -> dict[str, Any]:
    """Run the existing hybrid pipeline when configured; otherwise answer locally."""
    topic = str(topic or "").strip()

    # Offline model selected explicitly — route straight to local Ollama
    if isinstance(model_name, str) and model_name.startswith("local:"):
        ollama_model = model_name[len("local:"):]
        return _local_answer(
            topic,
            f"Offline mode: using local model {ollama_model}.",
            history=history,
            retrieve_local=True,
            local_model=ollama_model,
        )

    if requested_project_paths(topic):
        return _local_answer(
            topic,
            "The requested project file is handled from the local project directory; web research was skipped.",
            history=history,
            retrieve_local=False,
            local_model=local_model,
        )
    try:
        validate_model_configuration(model_name)
    except ModelConfigurationError:
        return _local_answer(
            topic,
            "OpenRouter is not configured; web research was skipped and the local model was used.",
            history=history,
            local_model=local_model,
        )

    try:
        result = run_research(topic, model_name)
        result["generation_mode"] = "api_research"
        result["message_type"] = "research"
        result["model_label"] = model_name or "OpenRouter"
        result["fallback_notice"] = None
        return result
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, (ModelConfigurationError, ResearchPipelineError)):
            reason = str(exc)
        else:
            reason = f"OpenRouter research failed ({error_type})"
        return _local_answer(
            topic,
            f"{reason} Local Ollama was used instead.",
            history=history,
            local_model=local_model,
        )
