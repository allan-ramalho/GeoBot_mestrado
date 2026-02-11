"""
Script de atualização da base RAG.

Use este script separado para indexar PDFs e atualizar o backend (ChromaDB/Supabase).
"""

import argparse
from pathlib import Path

from rag_engine import RAGEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atualizar base RAG do GeoBot")
    parser.add_argument("--backend", choices=["chroma", "supabase", "none"], default=None)
    parser.add_argument("--database-path", default="rag_database")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    rag = RAGEngine(
        database_path=Path(args.database_path),
        backend=args.backend,
        strict=True
    )
    rag.initialize()
    rag.index_documents(
        force_reindex=args.force_reindex,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        clear_existing=args.clear_existing
    )


if __name__ == "__main__":
    main()
