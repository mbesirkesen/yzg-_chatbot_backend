"""
RAG motoru: multilingual-e5-small ile embedding, Chroma benzerlik araması, LLM için bağlam üretimi.

E5 retrieval kuralı: sorgular \"query: \", dokümanlar \"passage: \" öneki ile embed edilir.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import chromadb
import torch
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings


def _e5_query_text(user_text: str) -> str:
    return f"query: {user_text.strip()}"


def _e5_passage_text(chunk_text: str) -> str:
    return f"passage: {chunk_text.strip()}"


def _normalize_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Chroma metadata değerlerini skalar türlere indirger."""
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


@dataclass
class RetrievedSource:
    """Tek bir kaynak satırı (API şemasına map edilir)."""

    id: str | None
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGRetrieveResult:
    """retrieve() çıktısı."""

    context: str
    sources: list[RetrievedSource]
    raw_documents: list[str]
    collection_empty: bool


class RAGEngine:
    """
    Chroma kalıcı istemcisi ve paylaşılan embedding modeli.
    Bloklayan encode/DB çağrıları async sarmalayıcılar üzerinden thread'de çalışır.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self._chroma: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None

    def _load_model_sync(self) -> SentenceTransformer:
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self._settings.embedding_model, device=device)
        return self._model

    def _get_collection_sync(self) -> Collection:
        if self._chroma is None:
            self._chroma = chromadb.PersistentClient(path=self._settings.chroma_persist_dir)
        if self._collection is None:
            self._collection = self._chroma.get_or_create_collection(
                name=self._settings.chroma_collection_name,
            )
        return self._collection

    def embed_query_sync(self, text: str) -> list[float]:
        model = self._load_model_sync()
        prefixed = _e5_query_text(text)
        vec = model.encode(
            [prefixed],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].tolist()

    def embed_passages_sync(self, texts: list[str]) -> list[list[float]]:
        """Ingestion ve toplu işlemler için passage önekli embedding."""
        model = self._load_model_sync()
        prefixed = [_e5_passage_text(t) for t in texts]
        vecs = model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        return [v.tolist() for v in vecs]

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query_sync, text)

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_passages_sync, texts)

    def _count_documents_sync(self) -> int:
        coll = self._get_collection_sync()
        return coll.count()

    async def collection_document_count(self) -> int:
        return await asyncio.to_thread(self._count_documents_sync)

    def _retrieve_sync(self, query: str, top_k: int) -> RAGRetrieveResult:
        coll = self._get_collection_sync()
        count = coll.count()
        if count == 0:
            return RAGRetrieveResult(
                context="",
                sources=[],
                raw_documents=[],
                collection_empty=True,
            )

        q_emb = self.embed_query_sync(query)
        n = min(top_k, max(1, count))
        result = coll.query(
            query_embeddings=[q_emb],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs = (result.get("documents") or [[]])[0] or []
        metas = (result.get("metadatas") or [[]])[0] or []
        ids = (result.get("ids") or [[]])[0] or []

        sources: list[RetrievedSource] = []
        lines: list[str] = []
        for i, doc in enumerate(docs):
            doc = doc or ""
            doc = re.sub(r'\bNone\b', '', doc)
            doc = re.sub(r'\s{2,}', ' ', doc).strip()
            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
            sid = ids[i] if i < len(ids) else None
            snippet = doc[:280] + ("…" if len(doc) > 280 else "")
            sources.append(
                RetrievedSource(
                    id=str(sid) if sid is not None else None,
                    snippet=snippet,
                    metadata=meta,
                )
            )
            meta_line = ", ".join(f"{k}={v}" for k, v in sorted(meta.items())[:8])
            block = f"[{i + 1}] {doc.strip()}"
            if meta_line:
                block += f"\n    (Kaynak: {meta_line})"
            lines.append(block)

        context = "\n\n".join(lines) if lines else ""
        return RAGRetrieveResult(
            context=context,
            sources=sources,
            raw_documents=list(docs),
            collection_empty=False,
        )

    async def retrieve(self, query: str, top_k: int) -> RAGRetrieveResult:
        """Asenkron: embedding + Chroma sorgusu thread pool'da."""
        return await asyncio.to_thread(self._retrieve_sync, query, top_k)

    def close(self) -> None:
        """Kaynakları serbest bırak (Chroma istemcisi hafif; model bellekte kalabilir)."""
        self._collection = None
        self._chroma = None
        # Modeli süreç sonuna kadar tutmak ilk istek gecikmesini önler; gerekirse None yapılabilir.
        # self._model = None
