# UYARI: Bu script BAAI/bge-m3 embedding kullanır. FastAPI RAG (app/) ise
# intfloat/multilingual-e5-small kullanır — aynı Chroma klasöründe ikisini
# karıştırmayın. Üretimde vektör yüklemek için: python -m app.ingestion_service <dosya.json>
# Bu script varsayılan olarak ayrı koleksiyon adı (turkish_recipes) kullanır; API turkish_cuisine okur.
#
# =============================================================================
# MODÜL 1: KURULUM KOMUTLARI
# =============================================================================
# Aşağıdaki komutları terminalden çalıştırarak gerekli kütüphaneleri kurun:
#
#   pip install langchain langchain-community langchain-chroma
#   pip install chromadb
#   pip install sentence-transformers
#   pip install huggingface-hub
#
# NOT: BAAI/bge-m3 modeli ilk çalıştırmada (~2.3 GB) otomatik indirilir.
# =============================================================================

import json
import os
from typing import Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =============================================================================
# MODÜL 2: VERİ YÜKLEME VE TEMİZLEME
# =============================================================================

def load_and_clean_recipes(filepath: str) -> list[dict]:
    """
    JSON dosyasını okur ve eksik (null) alanları temizler.

    Sayısal alanlar (porsiyon, süreler) None ise 0'a,
    liste alanları None ise boş listeye dönüştürülür.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        recipes: list[dict] = json.load(f)

    print(f"[OK] {len(recipes)} tarif yuklendi.")

    cleaned = []
    for recipe in recipes:
        # Sayısal null alanları 0 yap
        recipe["porsiyon"] = recipe.get("porsiyon") or 0
        recipe["hazirlik_suresi_dk"] = recipe.get("hazirlik_suresi_dk") or 0
        recipe["pisirme_suresi_dk"] = recipe.get("pisirme_suresi_dk") or 0

        # Liste alanları boşsa boş liste ata
        recipe["pisirme_yontemi"] = recipe.get("pisirme_yontemi") or []
        recipe["malzemeler"] = recipe.get("malzemeler") or []
        recipe["yapilis_adimlari"] = recipe.get("yapilis_adimlari") or []

        # Metin alanları boşsa "Belirtilmemiş" yap
        recipe["tarif_adi"] = recipe.get("tarif_adi") or "Belirtilmemiş"
        recipe["kategori"] = recipe.get("kategori") or "Belirtilmemiş"
        recipe["zorluk"] = recipe.get("zorluk") or "Belirtilmemiş"

        cleaned.append(recipe)

    print("[OK] Veri temizleme tamamlandi.")
    return cleaned


# =============================================================================
# MODÜL 3: SEMANTİK METİN VE METADATA OLUŞTURMA
# =============================================================================

def _format_ingredients(malzemeler: list[dict]) -> str:
    """Malzeme listesini 'miktar birim isim' formatında tek cümleye çevirir."""
    parts = []
    for m in malzemeler:
        miktar = m.get("miktar", "")
        birim = m.get("birim", "")
        isim = m.get("isim", "")
        parts.append(f"{miktar} {birim} {isim}".strip())
    return ", ".join(parts) if parts else "Belirtilmemiş"


def _format_steps(adimlar: list[str]) -> str:
    """Yapılış adımlarını numaralı liste olarak birleştirir."""
    if not adimlar:
        return "Belirtilmemiş"
    return " ".join(f"{i + 1}. {adim}" for i, adim in enumerate(adimlar))


def build_documents(recipes: list[dict]) -> list[Document]:
    """
    Her tarif için bir LangChain Document objesi üretir.

    page_content: Vektör aramasında kullanılacak akıcı Türkçe metin.
    metadata: Filtreleme için kategori, zorluk vb. etiketler.
    """
    documents = []

    for recipe in recipes:
        tarif_adi = recipe["tarif_adi"]
        kategori = recipe["kategori"]
        zorluk = recipe["zorluk"]
        pisirme_yontemi = ", ".join(recipe["pisirme_yontemi"]) or "Belirtilmemiş"
        malzemeler_str = _format_ingredients(recipe["malzemeler"])
        adimlar_str = _format_steps(recipe["yapilis_adimlari"])

        # Semantik arama için zengin ve akıcı Türkçe metin
        page_content = (
            f"Bu tarifin adı {tarif_adi}. "
            f"Bu bir {kategori} tarifidir ve zorluk derecesi {zorluk}. "
            f"Pişirme yöntemi: {pisirme_yontemi}. "
            f"Malzemeler: {malzemeler_str}. "
            f"Yapılış Adımları: {adimlar_str}"
        )

        # Filtreleme için düz metadata (ChromaDB yalnızca str/int/float/bool destekler)
        metadata: dict[str, Any] = {
            "tarif_adi": tarif_adi,
            "kategori": kategori,
            "zorluk": zorluk,
            "pisirme_yontemi": pisirme_yontemi,
            "porsiyon": int(recipe["porsiyon"]),
            "hazirlik_suresi_dk": int(recipe["hazirlik_suresi_dk"]),
            "pisirme_suresi_dk": int(recipe["pisirme_suresi_dk"]),
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"[OK] {len(documents)} Document objesi olusturuldu.")
    return documents


# =============================================================================
# MODÜL 4: EMBEDDING VE VEKTÖR VERİTABANI
# =============================================================================

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-m3 modelini CPU üzerinde yükler.
    Model ilk çalıştırmada HuggingFace Hub'dan indirilir (~2.3 GB).
    Türkçe dahil 100+ dili destekleyen çok dilli bir modeldir.
    """
    print("[...] Embedding modeli yukleniyor (ilk seferde indirme ~2.3 GB)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # Kosinüs benzerliği için gerekli
            "batch_size": 32,
        },
    )

    print("[OK] Embedding modeli hazir.")
    return embeddings


def build_vector_db(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = "./turkish_cuisine_db",
    batch_size: int = 500,
) -> Chroma:
    """
    Document listesini ChromaDB'ye batch'ler halinde ekler ve diske kaydeder.
    Büyük veri setlerinde bellek taşmasını önlemek için batch_size kullanılır.
    """
    print(f"[...] ChromaDB olusturuluyor -> {persist_dir}")
    os.makedirs(persist_dir, exist_ok=True)

    # İlk batch ile veritabanını başlat
    first_batch = documents[:batch_size]
    vectordb = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="turkish_recipes",
    )
    print(f"    [{batch_size}/{len(documents)}] eklendi...")

    # Kalan batch'leri ekle
    for start in range(batch_size, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        vectordb.add_documents(batch)
        added = min(start + batch_size, len(documents))
        print(f"    [{added}/{len(documents)}] eklendi...")

    print(f"[OK] Vektor veritabani kaydedildi: {persist_dir}")
    return vectordb


# =============================================================================
# ANA AKIŞ
# =============================================================================

def main():
    DATA_FILE = "recipes_groq_cleaned.json"
    DB_DIR = "./turkish_cuisine_db"

    try:
        # 1. Veri yükle ve temizle
        recipes = load_and_clean_recipes(DATA_FILE)

        # 2. Document objelerini oluştur
        documents = build_documents(recipes)

        # 3. Embedding modelini yükle
        embeddings = load_embedding_model()

        # 4. ChromaDB'ye kaydet
        vectordb = build_vector_db(documents, embeddings, persist_dir=DB_DIR)

        # 5. Hızlı doğrulama: örnek sorgu
        print("\n[TEST] Ornek sorgu: 'kolay mercimek corbasi tarifi'")
        results = vectordb.similarity_search("kolay mercimek corbasi tarifi", k=3)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.metadata['tarif_adi']} ({doc.metadata['kategori']}, {doc.metadata['zorluk']})")

        print("\n[OK] Tum islemler basariyla tamamlandi.")

    except FileNotFoundError as e:
        print(f"[HATA] Dosya bulunamadı: {e}")
    except json.JSONDecodeError as e:
        print(f"[HATA] JSON parse hatası: {e}")
    except Exception as e:
        print(f"[HATA] Beklenmeyen hata: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
