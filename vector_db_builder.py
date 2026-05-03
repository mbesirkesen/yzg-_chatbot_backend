"""
Tarif JSON'unu ChromaDB'ye yükler (ingestion_service sarmalayıcı).
Kullanım: python vector_db_builder.py [dosya.json]
Varsayılan dosya: recipes_groq_cleaned.json

Embedding: intfloat/multilingual-e5-small  (API ile aynı model)
Koleksiyon: turkish_cuisine                (API ile aynı isim)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion_service import ingest_path_sync


def main() -> None:
    data_file = sys.argv[1] if len(sys.argv) > 1 else "recipes_groq_cleaned.json"
    print(f"[...] Yükleniyor: {data_file}")
    try:
        result = ingest_path_sync(data_file)
        print(
            f"[OK] {result.num_chunks} chunk -> "
            f"koleksiyon '{result.collection}' ({result.persist_dir})"
        )
    except FileNotFoundError as e:
        print(f"[HATA] Dosya bulunamadı: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[HATA] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
