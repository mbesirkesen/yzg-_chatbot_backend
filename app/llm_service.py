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
    "Sen bir Türk mutfağı şefisin. "
    "Görevin yalnızca Türk mutfağı, yemek tarifleri, malzemeler ve pişirme teknikleri hakkında yardım etmektir. "
    "Kullanıcının sorusu bu konularla ilgili değilse, kibarca konuya yönlendir ve başka sorulara cevap verme. "
    "Her zaman yalnızca Türkçe yanıt ver, başka bir dil kullanma. "
    "Verilen döküman parçalarına dayan; döküman yoksa genel Türk mutfağı bilgini kullan. "
    "Yanıtlarını sıcak, samimi ve bilgilendirici bir üslupla ver.\n\n"
    "YANIT FORMATI — aşağıdaki kurallara kesinlikle uy:\n\n"
    "KURAL 1 — Döküman metnini asla olduğu gibi kopyalama. Bilgiyi kullanarak kendın yaz.\n"
    "KURAL 2 — Yanıtta seçenek listesini tekrarlama.\n"
    "KURAL 3 — Bir yanıtta yalnızca bir şey yap: ya liste ver, ya tarif ver.\n\n"
    "Kullanıcı genel tarif isterse (kategori: 'tavuklu tarif', 'tatlı öner' vb.):\n"
    "  Sadece numaralı liste: isim ve tek cümle açıklama. Son satır: 'Hangisini istersiniz?'\n"
    "  ASLA malzeme veya yapılış adımı yazma.\n\n"
    "Kullanıcı belirli bir tarif seçerse (isim veya numara), YALNIZCA şu şablonu kullan:\n\n"
    "### [Tarif Adı]\n\n"
    "🕐 **Süre:** [X dk] &nbsp;|&nbsp; 📊 **Zorluk:** [Kolay/Orta/Zor] &nbsp;|&nbsp; 🍳 **Yöntem:** [yöntem]\n\n"
    "---\n\n"
    "#### 🛒 Malzemeler\n\n"
    "- miktar malzeme\n\n"
    "---\n\n"
    "#### 👨‍🍳 Yapılış\n\n"
    "1. Adım\n\n"
    "Şablonun dışına çıkma. Ekstra cümle, tekrar veya liste ekleme."
)


@runtime_checkable
class LLMService(Protocol):
    """Tüm LLM sağlayıcıları bu arayüzü uygular."""

    async def generate(
        self, user_message: str, context: str, history: list[dict] | None = None
    ) -> str:
        """Kullanıcı sorusu, RAG bağlamı ve konuşma geçmişine göre asenkron metin üretir."""
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
        "Döküman bağlamı yok. Yanıtı yalnızca yukarıdaki konuşma geçmişine ve "
        "Türk mutfağı bilgine dayanarak ver.\n\n"
        f"Soru: {user_message.strip()}"
    )


class MockLLMService:
    """API anahtarı olmadan geliştirme ve test için sahte yanıt."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate(
        self, user_message: str, context: str, history: list[dict] | None = None
    ) -> str:
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

    async def generate(
        self, user_message: str, context: str, history: list[dict] | None = None
    ) -> str:
        if not self._settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY tanımlı değil.")

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_TURKISH_CHEF}]
        for h in (history or [])[-8:]:  # son 4 tur (8 mesaj)
            role = h.get("role", "")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": _build_user_content(user_message, context)})

        payload = {
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 1024,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._settings.groq_base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(60.0, connect=15.0)
        max_retries = 3
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(max_retries):
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", "10"))
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    raise RuntimeError(
                        f"Groq API istek limiti aşıldı ve {max_retries} deneme sonrası yanıt alınamadı. "
                        f"Lütfen birkaç dakika bekleyip tekrar deneyin."
                    )
                response.raise_for_status()
                data = response.json()
                break

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq yanıtında choices bulunamadı.")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Groq boş içerik döndü.")
        # Tekrar döngüsü koruması: aynı cümle 3'ten fazla geçiyorsa kes
        lines = content.split('\n')
        seen: dict[str, int] = {}
        clean_lines = []
        for line in lines:
            key = line.strip()
            if key:
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 2:
                    break
            clean_lines.append(line)
        return '\n'.join(clean_lines).strip()


class ClaudeLLMService:
    """Gelecek: Anthropic Claude entegrasyonu."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate(
        self, user_message: str, context: str, history: list[dict] | None = None
    ) -> str:
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
