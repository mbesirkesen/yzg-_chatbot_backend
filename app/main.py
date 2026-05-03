"""
FastAPI uygulaması: CORS, /chat (RAG + LLM), /health.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.llm_service import get_llm_service
from app.rag_engine import RAGEngine, RAGRetrieveResult
from app.schemas import ChatRequest, ChatResponse, SourceItem

_ORDINAL_WORDS = {
    "ilk", "birinci", "ikinci", "üçüncü", "dördüncü", "beşinci",
    "altıncı", "yedinci", "sekizinci", "dokuzuncu", "onuncu",
    "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
}


def _is_reference_query(msg: str) -> bool:
    """Sıralı/işaret referansı içeren kısa sorgu mu? (RAG atlanır)"""
    words = msg.lower().split()
    return len(words) <= 5 and any(w in _ORDINAL_WORDS for w in words)


def _is_list_selection(msg: str, history: list) -> bool:
    """Önceki bot mesajında liste varsa ve kullanıcı kısa bir seçim yaptıysa RAG'ı atla."""
    if len(msg.split()) > 7 or '?' in msg:
        return False
    for h in reversed(history):
        if h.role == "assistant":
            return "Hangisini istersiniz" in h.content
    return False


limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])


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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Kullanıcı mesajını alır, Chroma'dan bağlam getirir, LLM ile yanıt üretir.
    """
    msg = body.message.strip()
    if len(msg) < 5 or len(msg.split()) < 2:
        return ChatResponse(
            answer="Lütfen sorunuzu daha ayrıntılı belirtin. "
                   "Örneğin: \"Tavuklu yemek tarifi ver\" veya \"Kolay bir tatlı önerir misin?\"",
        )

    rag: RAGEngine = app.state.rag
    llm = app.state.llm
    settings = app.state.settings

    top_k = body.top_k if body.top_k is not None else settings.default_top_k

    try:
        skip_rag = (
            (_is_reference_query(msg) and body.history) or
            _is_list_selection(msg, body.history)
        )
        if skip_rag:
            rag_result = RAGRetrieveResult(
                context="", sources=[], raw_documents=[], collection_empty=False
            )
        else:
            rag_result = await rag.retrieve(msg, top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG hatası: {e}") from e

    warning: str | None = None
    if rag_result.collection_empty:
        warning = (
            "Vektör koleksiyonu boş veya henüz veri yüklenmemiş. "
            "Önce ingestion ile veri ekleyin; yanıt mock/genel modda üretilebilir."
        )

    try:
        history = [{"role": h.role, "content": h.content} for h in body.history]
        answer = await llm.generate(body.message, rag_result.context, history)
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
