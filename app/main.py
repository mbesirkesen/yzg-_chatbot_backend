"""
FastAPI uygulaması: CORS, /chat (RAG + LLM), /health.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import get_settings
from app.llm_service import get_llm_service
from app.rag_engine import RAGEngine
from app.schemas import ChatRequest, ChatResponse, SourceItem


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama ömrü: RAG motoru ve LLM servis örnekleri."""
    settings = get_settings()
    rag = RAGEngine(settings)
    llm = get_llm_service(settings)
    app.state.settings = settings
    app.state.rag = rag
    app.state.llm = llm
    yield
    rag.close()


app = FastAPI(
    title="Türk Mutfağı RAG Chatbot API",
    description="RAG tabanlı Türk mutfağı asistanı backend'i.",
    lifespan=lifespan,
)

_settings_for_cors = get_settings()
_origins = _settings_for_cors.cors_origins_list()
_allow_credentials = "*" not in _origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, Any]:
    """Tarayıcıda / açıldığında API özeti (404 yerine)."""
    return {
        "service": app.title,
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
        "hint": "Sohbet için POST /chat ve JSON gövde: {\"message\": \"...\"}",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Tarayıcının otomatik favicon isteğini 404 logu üretmeden sonlandırır."""
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Basit sağlık kontrolü ve koleksiyondaki döküman sayısı."""
    rag: RAGEngine = app.state.rag
    settings = app.state.settings
    try:
        count = await rag.collection_document_count()
    except Exception as e:  # noqa: BLE001
        return {
            "status": "degraded",
            "error": str(e),
            "chroma_collection": settings.chroma_collection_name,
        }
    return {
        "status": "ok",
        "chroma_collection": settings.chroma_collection_name,
        "document_count": count,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """
    Kullanıcı mesajını alır, Chroma'dan bağlam getirir, LLM ile yanıt üretir.
    """
    rag: RAGEngine = app.state.rag
    llm = app.state.llm
    settings = app.state.settings

    top_k = body.top_k if body.top_k is not None else settings.default_top_k

    try:
        rag_result = await rag.retrieve(body.message.strip(), top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG hatası: {e}") from e

    warning: str | None = None
    if rag_result.collection_empty:
        warning = (
            "Vektör koleksiyonu boş veya henüz veri yüklenmemiş. "
            "Önce ingestion ile veri ekleyin; yanıt mock/genel modda üretilebilir."
        )

    try:
        answer = await llm.generate(body.message, rag_result.context)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LLM hatası: {e}") from e

    sources = [
        SourceItem(
            id=s.id,
            snippet=s.snippet,
            metadata=s.metadata,
        )
        for s in rag_result.sources
    ]

    retrieved = rag_result.raw_documents if body.include_retrieved_chunks else None

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved,
        warning=warning,
    )
