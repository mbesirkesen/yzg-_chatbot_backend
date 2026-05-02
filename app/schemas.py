"""
API istek/yanıt şemaları (Pydantic).
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat gövdesi."""

    message: str = Field(..., min_length=1, description="Kullanıcı sorusu veya mesajı")
    session_id: str | None = Field(default=None, description="İsteğe bağlı oturum kimliği")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Getirilecek benzer chunk sayısı; boşsa sunucu varsayılanı",
    )
    include_retrieved_chunks: bool = Field(
        default=False,
        description="True ise yanıtta ham alınan parçalar da döner (debug)",
    )


class SourceItem(BaseModel):
    """Kaynak özeti (metadata + kısa metin)."""

    id: str | None = None
    snippet: str = Field(..., description="İlgili metnin kısa özeti veya başı")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """POST /chat yanıtı."""

    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    retrieved_chunks: list[str] | None = Field(
        default=None,
        description="include_retrieved_chunks=True ise dolu",
    )
    warning: str | None = Field(
        default=None,
        description="Örn. vektör veritabanı boşsa bilgilendirme",
    )
