"""
LLM soyutlaması: şimdilik Mock; Groq/Claude eklendiğinde aynı arayüz kullanılır.
"""

from __future__ import annotations

import asyncio
import httpx
from typing import Protocol, runtime_checkable

from app.config import Settings, get_settings

# Sistem talimatı (sabit)
SYSTEM_PROMPT_TURKISH_CHEF = (
    "Sen bir Türk mutfağı şefisin. Verilen dökümanlara göre samimi ve bilgilendirici bir "
    "dilde cevap ver."
)


@runtime_checkable
class LLMService(Protocol):
    """Tüm LLM sağlayıcıları bu arayüzü uygular."""

    async def generate(self, user_message: str, context: str) -> str:
        """Kullanıcı sorusu ve RAG bağlamına göre asenkron metin üretir."""
        ...


def _build_user_content(user_message: str, context: str) -> str:
    """Gerçek API entegrasyonunda system / user ayrımı için kullanıcı tarafı metni."""
    if context.strip():
        return (
            "Aşağıdaki döküman parçalarını kullanarak soruyu yanıtla.\n\n"
            f"Dökümanlar:\n{context}\n\n"
            f"Soru: {user_message.strip()}"
        )
    return (
        "Veritabanında henüz ilgili döküm bulunamadı veya koleksiyon boş. "
        "Bunu nazikçe belirterek genel Türk mutfağı bilgine dayalı kısa bir yanıt ver.\n\n"
        f"Soru: {user_message.strip()}"
    )


class MockLLMService:
    """API anahtarı olmadan geliştirme ve test için sahte yanıt."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate(self, user_message: str, context: str) -> str:
        # Senkron iş yok; küçük gecikme ile gerçek async davranışı taklit edilebilir
        await asyncio.sleep(0)

        user_block = _build_user_content(user_message, context)
        if not context.strip():
            return (
                "[Mock LLM] Merhaba! Şu an vektör veritabanında arama sonucu yok veya veri "
                "yüklenmemiş. Lütfen önce ingestion ile Chroma koleksiyonunu doldurun. "
                f"Sorduğunuz konu: «{user_message.strip()[:120]}»"
            )

        preview = context.strip().split("\n\n")[0][:400]
        return (
            "[Mock LLM] (Gerçek API bağlandığında bu metin model çıktısı olacak.) "
            "Şef gözüyle kısaca: dökümanlarda şuna benzer bilgiler bulundu — "
            f"{preview}"
            f"{'…' if len(context) > 400 else ''} "
            "Detaylı tarif ve teknik için yukarıdaki kaynak parçalarına güvenebilirsiniz."
        )


class GroqLLMService:
    """Groq OpenAI-uyumlu endpoint ile gerçek yanıt üretir."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate(self, user_message: str, context: str) -> str:
        if not self._settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY tanımlı değil.")

        payload = {
            "model": self._settings.groq_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TURKISH_CHEF},
                {"role": "user", "content": _build_user_content(user_message, context)},
            ],
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._settings.groq_base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(45.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq yanıtında choices bulunamadı.")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Groq boş içerik döndü.")
        return content


class ClaudeLLMService:
    """Gelecek: Anthropic Claude entegrasyonu."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate(self, user_message: str, context: str) -> str:
        raise NotImplementedError("ClaudeLLMService henüz uygulanmadı; LLM_PROVIDER=mock kullanın.")


def get_llm_service(settings: Settings | None = None) -> LLMService:
    """
    Ortam değişkenine göre LLM sağlayıcısı döndürür.
    Şimdilik yalnızca mock üretir; groq/anthropic seçilirse NotImplemented hatası verir.
    """
    s = settings or get_settings()
    provider = (s.llm_provider or "mock").lower().strip()
    if provider == "mock":
        return MockLLMService(s)
    if provider == "groq":
        return GroqLLMService(s)
    if provider in ("anthropic", "claude"):
        return ClaudeLLMService(s)
    return MockLLMService(s)


__all__ = [
    "SYSTEM_PROMPT_TURKISH_CHEF",
    "LLMService",
    "MockLLMService",
    "GroqLLMService",
    "ClaudeLLMService",
    "get_llm_service",
    "_build_user_content",
]
