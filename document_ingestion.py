"""Local document extraction and stable overlapping text chunking."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TEXT_CHARS = 1_000_000
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150
DOCX_XML_LIMIT_MULTIPLIER = 20


class DocumentIngestionError(ValueError):
    """A safe, descriptive document-ingestion failure."""


@dataclass(frozen=True)
class IngestionConfig:
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self):
        if self.max_file_bytes <= 0 or self.max_text_chars <= 0 or self.chunk_size <= 0:
            raise ValueError("File, text, and chunk size limits must be positive.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source_filename: str
    page_number: int | None
    document_id: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    source_filename: str
    file_type: str
    content_hash: str
    document_metadata: dict[str, Any]
    chunks: tuple[DocumentChunk, ...]


def _extract_pdf(payload: bytes, max_text_chars: int) -> list[tuple[int | None, str]]:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentIngestionError("PDF extraction requires the existing PyMuPDF dependency.") from exc
    try:
        with fitz.open(stream=payload, filetype="pdf") as pdf:
            pages = []
            total = 0
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                total += len(text)
                if total > max_text_chars:
                    raise DocumentIngestionError("Extracted text exceeds the configured text-length limit.")
                if text.strip():
                    pages.append((page_index, text))
            return pages
    except DocumentIngestionError:
        raise
    except Exception as exc:
        raise DocumentIngestionError(f"Could not read the PDF ({type(exc).__name__}).") from exc


def _extract_docx(payload: bytes, max_text_chars: int) -> list[tuple[int | None, str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if sum(item.file_size for item in members) > max_text_chars * DOCX_XML_LIMIT_MULTIPLIER:
                raise DocumentIngestionError("DOCX contents exceed the configured extraction limit.")
            try:
                xml = archive.read("word/document.xml")
            except KeyError as exc:
                raise DocumentIngestionError("The DOCX file has no readable document body.") from exc
    except DocumentIngestionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise DocumentIngestionError(f"Could not read the DOCX ({type(exc).__name__}).") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise DocumentIngestionError("The DOCX document body contains invalid XML.") from exc
    paragraphs = []
    total = 0
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = "".join(
            node.text or ""
            for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        )
        if text.strip():
            total += len(text)
            if total > max_text_chars:
                raise DocumentIngestionError("Extracted text exceeds the configured text-length limit.")
            paragraphs.append(text)
    return [(None, "\n".join(paragraphs))] if paragraphs else []


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into reproducible, whitespace-aware overlapping chunks."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    cleaned = text.strip()
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        next_start = max(start + 1, end - chunk_overlap)
        while next_start < end and cleaned[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def _validated_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    reserved = {"source_filename", "file_type", "document_id", "chunk_id", "page_number"}
    result = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or key in reserved:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return result


def ingest_document(path: str | Path, *, config: IngestionConfig | None = None,
                    metadata: Mapping[str, Any] | None = None) -> IngestedDocument:
    """Extract a supported local document and return stable source-aware chunks."""
    config = config or IngestionConfig()
    file_path = Path(path)
    if file_path.name.casefold() == ".env" or file_path.name.casefold().startswith(".env."):
        raise DocumentIngestionError("Credential files are not accepted by the knowledge base.")
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentIngestionError(f"Unsupported file type '{extension or '(none)'}'. Supported: PDF, TXT, Markdown, DOCX.")
    try:
        if file_path.stat().st_size > config.max_file_bytes:
            raise DocumentIngestionError("File exceeds the configured file-size limit.")
        payload = file_path.read_bytes()
    except DocumentIngestionError:
        raise
    except OSError as exc:
        raise DocumentIngestionError(f"Could not read the document ({type(exc).__name__}).") from exc
    if len(payload) > config.max_file_bytes:
        raise DocumentIngestionError("File exceeds the configured file-size limit.")

    if extension == ".pdf":
        pages = _extract_pdf(payload, config.max_text_chars)
    elif extension == ".docx":
        pages = _extract_docx(payload, config.max_text_chars)
    else:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentIngestionError("Text documents must be UTF-8 encoded.") from exc
        if len(text) > config.max_text_chars:
            raise DocumentIngestionError("Extracted text exceeds the configured text-length limit.")
        pages = [(None, text)]

    if not any(text.strip() for _, text in pages):
        if extension == ".pdf":
            raise DocumentIngestionError(
                "No selectable text was found in this PDF. Scanned PDFs may require OCR; OCR is out of scope for this phase."
            )
        raise DocumentIngestionError("The document contains no extractable text.")

    content_hash = hashlib.sha256(payload).hexdigest()
    source_filename = file_path.name
    document_id = hashlib.sha256(f"{source_filename.casefold()}:{content_hash}".encode("utf-8")).hexdigest()
    document_metadata = _validated_metadata(metadata)
    chunks = []
    chunk_index = 0
    for page_number, text in pages:
        for chunk_text in split_text(text, config.chunk_size, config.chunk_overlap):
            chunk_index += 1
            chunk_id = hashlib.sha256(
                f"{document_id}:{page_number}:{chunk_index}:{chunk_text}".encode("utf-8")
            ).hexdigest()
            chunk_meta = {
                **document_metadata,
                "source_filename": source_filename,
                "file_type": extension.lstrip("."),
                "document_id": document_id,
                "chunk_id": chunk_id,
                "page_number": page_number,
            }
            chunks.append(DocumentChunk(
                text=chunk_text,
                source_filename=source_filename,
                page_number=page_number,
                document_id=document_id,
                chunk_id=chunk_id,
                metadata=chunk_meta,
            ))

    return IngestedDocument(
        document_id=document_id,
        source_filename=source_filename,
        file_type=extension.lstrip("."),
        content_hash=content_hash,
        document_metadata=document_metadata,
        chunks=tuple(chunks),
    )
