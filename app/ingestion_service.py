"""
Chunk'lanmış JSON/CSV verisini okuyup ChromaDB'ye yazar (E5 passage embedding ile).
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.config import Settings, get_settings
from app.rag_engine import RAGEngine, _normalize_metadata

logger = logging.getLogger(__name__)

TEXT_KEYS = ("text", "content", "chunk", "body")

# Tarif şemasından gelen büyük alanlar metadata'ya yazılmaz (Chroma sınırı + gürültü).
_RECIPE_META_SKIP = frozenset({"malzemeler", "yapilis_adimlari"})


@dataclass
class IngestResult:
    """Ingestion özeti."""

    path: str
    num_chunks: int
    collection: str
    persist_dir: str


def _text_key_in_row(row: dict[str, Any]) -> str | None:
    """Satırda metin alanı anahtarını bulur (büyük/küçük harf duyarsız)."""
    lower = {str(k).lower(): k for k in row}
    for candidate in TEXT_KEYS:
        if candidate in lower:
            return lower[candidate]
    return None


def _extract_text(row: dict[str, Any]) -> str | None:
    key = _text_key_in_row(row)
    if key is None:
        return None
    v = row.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _row_metadata(row: dict[str, Any], exclude_keys: set[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for k, v in row.items():
        if k in exclude_keys or str(k).lower() == "metadata":
            continue
        if v is not None:
            meta[str(k)] = v
    extra = row.get("metadata")
    if isinstance(extra, dict):
        for k, v in extra.items():
            if v is not None:
                meta[str(k)] = v
    return _normalize_metadata(meta)


def _stable_id(raw_id: str | None, text: str, index: int) -> str:
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id).strip()
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"chunk_{index}_{h}"


def _format_ingredients(malzemeler: Any) -> str:
    """recipes_groq_cleaned.json içindeki malzeme listesini metne çevirir."""
    if not isinstance(malzemeler, list):
        return "Belirtilmemiş"
    parts: list[str] = []
    for m in malzemeler:
        if not isinstance(m, dict):
            continue
        miktar = m.get("miktar", "")
        birim = m.get("birim", "")
        isim = m.get("isim", "")
        parts.append(f"{miktar} {birim} {isim}".strip())
    return ", ".join(parts) if parts else "Belirtilmemiş"


def _format_steps(adimlar: Any) -> str:
    if not isinstance(adimlar, list) or not adimlar:
        return "Belirtilmemiş"
    return " ".join(f"{i + 1}. {adim}" for i, adim in enumerate(adimlar))


def _recipe_page_content(recipe: dict[str, Any]) -> str:
    """
    vector_db_builder ile aynı fikir: semantik arama için tek parça Türkçe metin.
    (Embedding RAGEngine'de passage: öneki ile yapılır.)
    """
    tarif_adi = recipe.get("tarif_adi") or "Belirtilmemiş"
    kategori = recipe.get("kategori") or "Belirtilmemiş"
    zorluk = recipe.get("zorluk") or "Belirtilmemiş"
    py = recipe.get("pisirme_yontemi")
    if isinstance(py, list):
        pisirme_yontemi = ", ".join(str(x) for x in py) or "Belirtilmemiş"
    elif py is None:
        pisirme_yontemi = "Belirtilmemiş"
    else:
        pisirme_yontemi = str(py)
    malzemeler_str = _format_ingredients(recipe.get("malzemeler"))
    adimlar_str = _format_steps(recipe.get("yapilis_adimlari"))
    return (
        f"Bu tarifin adı {tarif_adi}. "
        f"Bu bir {kategori} tarifidir ve zorluk derecesi {zorluk}. "
        f"Pişirme yöntemi: {pisirme_yontemi}. "
        f"Malzemeler: {malzemeler_str}. "
        f"Yapılış Adımları: {adimlar_str}"
    )


def _looks_like_recipe_without_text(row: dict[str, Any]) -> bool:
    """Kök liste formatındaki tarif kaydı (text/content yok, tarif_adi var)."""
    if row.get("tarif_adi") is None:
        return False
    return _text_key_in_row(row) is None


def _expand_recipe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tarif JSON'unu ingestion'ın beklediği şekilde text alanı ekler."""
    out: list[dict[str, Any]] = []
    for row in records:
        if _looks_like_recipe_without_text(row):
            out.append({**row, "text": _recipe_page_content(row)})
        else:
            out.append(row)
    return out


def _load_records_from_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict) and "chunks" in raw:
        inner = raw["chunks"]
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
    raise ValueError(
        "JSON: kök bir nesne listesi veya {\"chunks\": [...]} yapısı bekleniyor.",
    )


def _load_records_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV: başlık satırı bulunamadı.")
        lower = {h.lower(): h for h in reader.fieldnames}
        text_col = None
        for candidate in TEXT_KEYS:
            if candidate in lower:
                text_col = lower[candidate]
                break
        if text_col is None:
            raise ValueError(
                f"CSV: metin sütunu gerekli; şunlardan biri olmalı (büyük/küçük harf duyarsız): "
                f"{', '.join(TEXT_KEYS)}",
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            if not row:
                continue
            rows.append(dict(row))
        return rows


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Dosya uzantısına göre JSON veya CSV kayıtlarını yükler."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        records = _load_records_from_json(p)
        return _expand_recipe_records(records)
    if suffix == ".csv":
        return _load_records_from_csv(p)
    raise ValueError(f"Desteklenen uzantılar: .json, .csv (verilen: {suffix})")


def ingest_path_sync(
    path: str | Path,
    settings: Settings | None = None,
) -> IngestResult:
    """
    Senkron ingestion: kayıtları okur, embed eder, Chroma'ya upsert eder.
    Doküman metni ham saklanır; embedding RAGEngine içinde passage: öneki ile üretilir.
    """
    s = settings or get_settings()
    records = load_records(path)
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, row in enumerate(records):
        text = _extract_text(row)
        if not text:
            logger.warning("Satır %s atlandı: metin alanı yok.", i)
            continue
        text_col = _text_key_in_row(row)
        exclude = (
            {str(k) for k in TEXT_KEYS}
            | {"id", "ID", "Id"}
            | set(_RECIPE_META_SKIP)
        )
        if text_col:
            exclude.add(text_col)
        rid = row.get("id") or row.get("ID")
        ids.append(_stable_id(str(rid) if rid is not None else None, text, i))
        texts.append(text)
        metadatas.append(_row_metadata(row, exclude))

    if not texts:
        raise ValueError("Yüklenecek geçerli chunk yok (tüm satırlar boş mu?).")

    engine = RAGEngine(s)
    embeddings = engine.embed_passages_sync(texts)

    client = chromadb.PersistentClient(path=s.chroma_persist_dir)
    collection = client.get_or_create_collection(name=s.chroma_collection_name)
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    return IngestResult(
        path=str(Path(path).resolve()),
        num_chunks=len(texts),
        collection=s.chroma_collection_name,
        persist_dir=s.chroma_persist_dir,
    )


async def ingest_path(path: str | Path, settings: Settings | None = None) -> IngestResult:
    """Bloklamayı önlemek için ingestion'ı thread'de çalıştırır."""
    return await asyncio.to_thread(ingest_path_sync, path, settings)


def main() -> None:
    """Örnek: python -m app.ingestion_service veri.json"""
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Kullanım: python -m app.ingestion_service <dosya.json|dosya.csv>", file=sys.stderr)
        sys.exit(1)
    result = ingest_path_sync(sys.argv[1])
    print(
        f"Tamam: {result.num_chunks} chunk -> koleksiyon '{result.collection}' "
        f"({result.persist_dir})",
    )


if __name__ == "__main__":
    main()
