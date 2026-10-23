"""Small, testable operations used by the local document-library UI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from document_ingestion import DEFAULT_MAX_FILE_BYTES, DocumentIngestionError, SUPPORTED_EXTENSIONS


def ingest_uploaded_document(
    knowledge_base: Any,
    filename: str,
    payload: bytes,
    *,
    temporary_root: str | Path | None = None,
) -> dict[str, Any]:
    """Ingest one explicitly submitted upload through the existing knowledge-base API."""
    safe_name = Path(filename).name
    if not filename or safe_name != filename or safe_name in {".", ".."}:
        raise DocumentIngestionError("The uploaded filename is invalid.")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentIngestionError(
            f"Unsupported file type '{suffix or '(none)'}'. Supported: PDF, TXT, Markdown, DOCX."
        )
    if len(payload) > DEFAULT_MAX_FILE_BYTES:
        raise DocumentIngestionError("File exceeds the configured file-size limit (20 MiB).")

    # Temporary upload material is removed immediately after ingestion; only chunk
    # text and metadata are persisted by the local knowledge base.
    with tempfile.TemporaryDirectory(
        prefix="researchmind_upload_",
        dir=str(temporary_root) if temporary_root is not None else None,
    ) as temporary_directory:
        path = Path(temporary_directory) / safe_name
        path.write_bytes(payload)
        return knowledge_base.ingest_document(path)


def retrieve_local_documents(knowledge_base: Any, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Run local semantic retrieval only; this function has no provider dependency."""
    if not isinstance(query, str) or not query.strip():
        return []
    return knowledge_base.retrieve(query.strip(), top_k=top_k)


def delete_local_document(knowledge_base: Any, document_id: str, *, confirmed: bool) -> dict[str, Any] | None:
    """Delete exactly one selected document only after explicit confirmation."""
    if not confirmed:
        return None
    return knowledge_base.delete_document(document_id)


def supported_upload_extensions() -> list[str]:
    """Return the UI extension list in a stable display order."""
    order = (".pdf", ".txt", ".md", ".docx")
    return [extension[1:] for extension in order if extension in SUPPORTED_EXTENSIONS]
