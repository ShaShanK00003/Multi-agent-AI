"""Persistent local Chroma knowledge base with CPU-only embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from document_ingestion import IngestionConfig, ingest_document


PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PERSIST_DIRECTORY = PROJECT_DATA_DIR / "knowledge_base"
DEFAULT_MODEL_CACHE_DIRECTORY = PROJECT_DATA_DIR / "models" / "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "researchmind_documents"


class KnowledgeBaseError(RuntimeError):
    """A safe and actionable local knowledge-base failure."""


class _SentenceTransformerEncoder:
    """Lazily load the selected embedding model on CPU and cache it locally."""

    def __init__(self, model_name: str, cache_directory: Path):
        self.model_name = model_name
        self.cache_directory = cache_directory
        self._model = None

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.cache_directory.mkdir(parents=True, exist_ok=True)
                self._model = SentenceTransformer(
                    self.model_name,
                    device="cpu",
                    cache_folder=str(self.cache_directory),
                )
            except Exception as exc:
                raise KnowledgeBaseError(
                    f"Could not load embedding model '{self.model_name}' ({type(exc).__name__}). "
                    "Check that the model is cached locally or that it can be downloaded."
                ) from None
        try:
            vectors = self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as exc:
            raise KnowledgeBaseError(f"Embedding generation failed ({type(exc).__name__}).") from None


class KnowledgeBase:
    """A persistent Chroma collection with injectable client and embedder."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_function: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        client: Any | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.persist_directory = Path(persist_directory or DEFAULT_PERSIST_DIRECTORY).resolve()
        self.model_name = model_name
        self._embedding_function = embedding_function or _SentenceTransformerEncoder(
            model_name, DEFAULT_MODEL_CACHE_DIRECTORY
        )
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise KnowledgeBaseError(
                    "Chroma is not installed. Install the approved chromadb dependency to initialize the local knowledge base."
                ) from exc
            try:
                client = chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as exc:
                raise KnowledgeBaseError(
                    f"Could not open the local vector database at '{self.persist_directory}' "
                    f"({type(exc).__name__}). Existing database files were left untouched."
                ) from None
        self.client = client
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Could not open the Chroma collection '{collection_name}' ({type(exc).__name__}). "
                "Existing database files were left untouched."
            ) from None

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            vectors = self._embedding_function(texts)
            normalized = [[float(value) for value in vector] for vector in vectors]
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(f"Embedding generation failed ({type(exc).__name__}).") from None
        if len(normalized) != len(texts) or any(not vector for vector in normalized):
            raise KnowledgeBaseError("The embedding function returned an invalid number of vectors or an empty vector.")
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) != 1:
            raise KnowledgeBaseError("The embedding function returned vectors with inconsistent dimensions.")
        return normalized

    @staticmethod
    def _result_ids(result: dict[str, Any] | None) -> list[str]:
        return list((result or {}).get("ids") or [])

    def ingest_document(
        self,
        path: str | Path,
        *,
        config: IngestionConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract and upsert one document. Stable IDs make identical re-ingestion idempotent."""
        document = ingest_document(path, config=config, metadata=metadata)
        if not document.chunks:
            raise KnowledgeBaseError("Document extraction produced no chunks.")
        chunk_texts = [chunk.text for chunk in document.chunks]
        embeddings = self._embed(chunk_texts)
        chunk_ids = [chunk.chunk_id for chunk in document.chunks]
        try:
            prior = self.collection.get(
                where={"document_id": document.document_id},
                include=["metadatas"],
            )
            existing_ids = set(self._result_ids(prior))
            metadata_records = []
            for chunk in document.chunks:
                # Chroma metadata values must be scalar; structured metadata is preserved as JSON.
                safe_metadata = {
                    key: value for key, value in chunk.metadata.items()
                    if value is not None and isinstance(value, (str, int, float, bool))
                }
                safe_metadata["document_metadata_json"] = json.dumps(
                    document.document_metadata, ensure_ascii=False, sort_keys=True
                )
                metadata_records.append(safe_metadata)

            self.collection.upsert(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=metadata_records,
                embeddings=embeddings,
            )
            stale_ids = existing_ids.difference(chunk_ids)
            if stale_ids:
                self.collection.delete(ids=sorted(stale_ids))
        except Exception as exc:
            raise KnowledgeBaseError(f"Could not store document chunks ({type(exc).__name__}).") from None
        return {
            "document_id": document.document_id,
            "source_filename": document.source_filename,
            "file_type": document.file_type,
            "chunk_count": len(document.chunks),
            "added_chunks": len(set(chunk_ids).difference(existing_ids)),
            "duplicate": bool(existing_ids) and set(chunk_ids).issubset(existing_ids),
            "metadata": document.document_metadata,
        }

    def retrieve(self, query: str, *, top_k: int = 5,
                 min_similarity: float | None = None) -> list[dict[str, Any]]:
        """Return nearest source-aware chunks, with cosine distance and similarity."""
        if not isinstance(query, str) or not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        try:
            vector = self._embed([query.strip()])[0]
            response = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(f"Semantic retrieval failed ({type(exc).__name__}).") from None

        documents = (response.get("documents") or [[]])[0] or []
        metadatas = (response.get("metadatas") or [[]])[0] or []
        distances = (response.get("distances") or [[]])[0] or []
        records = []
        for index, text in enumerate(documents):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) and distances[index] is not None else None
            similarity = 1.0 - distance if distance is not None else None
            if min_similarity is not None and (similarity is None or similarity < min_similarity):
                continue
            raw_doc_metadata = metadata.pop("document_metadata_json", "{}")
            try:
                doc_metadata = json.loads(raw_doc_metadata)
            except (TypeError, json.JSONDecodeError):
                doc_metadata = {}
            page_number = metadata.get("page_number")
            records.append({
                "text": text,
                "source": metadata.get("source_filename", ""),
                "page": int(page_number) if page_number is not None else None,
                "document_id": metadata.get("document_id"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": distance,
                "similarity": similarity,
                "distance_metric": "cosine",
                "metadata": {**doc_metadata, **metadata},
            })
        return records

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete all chunks for one explicitly selected document ID."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("A document ID is required.")
        try:
            current = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            ids = self._result_ids(current)
            if ids:
                self.collection.delete(ids=ids)
        except Exception as exc:
            raise KnowledgeBaseError(f"Could not delete document chunks ({type(exc).__name__}).") from None
        return {"document_id": document_id, "deleted_chunks": len(ids), "deleted": bool(ids)}

    def list_documents(self) -> list[dict[str, Any]]:
        """List indexed documents from persisted chunk metadata, without passage text."""
        try:
            stored = self.collection.get(include=["metadatas"])
        except Exception as exc:
            raise KnowledgeBaseError(f"Could not list indexed documents ({type(exc).__name__}).") from None

        documents: dict[str, dict[str, Any]] = {}
        for metadata in stored.get("metadatas") or []:
            if not isinstance(metadata, dict):
                continue
            document_id = metadata.get("document_id")
            filename = metadata.get("source_filename")
            if not isinstance(document_id, str) or not document_id or not isinstance(filename, str) or not filename:
                continue
            record = documents.setdefault(document_id, {
                "document_id": document_id,
                "source_filename": filename,
                "file_type": metadata.get("file_type") if isinstance(metadata.get("file_type"), str) else "",
                "chunk_ids": set(),
            })
            chunk_id = metadata.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                record["chunk_ids"].add(chunk_id)

        result = []
        for record in documents.values():
            result.append({
                "document_id": record["document_id"],
                "source_filename": record["source_filename"],
                "file_type": record["file_type"],
                "chunk_count": len(record["chunk_ids"]),
            })
        return sorted(result, key=lambda item: (item["source_filename"].casefold(), item["document_id"]))
