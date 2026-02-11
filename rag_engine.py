"""
RAG Engine - Backend unificado (ChromaDB ou Supabase)
====================================================

Este módulo isola o motor de RAG para ser usado tanto pelo GeoBot
quanto por scripts de atualização da base.
"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover
    chromadb = None
    Settings = None

try:
    from supabase import create_client  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    create_client = None


@dataclass
class RAGConfig:
    database_path: Path = Path("rag_database")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: str = "chroma"  # chroma | supabase | none
    collection_name: str = "geobot_papers"
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_table: str = "rag_documents"
    supabase_rpc: str = "match_rag_documents"
    strict: bool = False


class RAGEngine:
    """
    Motor de Retrieval-Augmented Generation para contexto científico.

    Suporta dois backends:
    - ChromaDB (local, persistente)
    - Supabase (PostgreSQL + pgvector)
    """

    def __init__(
        self,
        database_path: Union[str, Path] = "rag_database",
        backend: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        collection_name: str = "geobot_papers",
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_table: str = "rag_documents",
        supabase_rpc: str = "match_rag_documents",
        strict: bool = False,
    ):
        env_backend = os.getenv("RAG_BACKEND")
        auto_backend = "supabase" if os.getenv("SUPABASE_URL") else "chroma"
        self.config = RAGConfig(
            database_path=Path(database_path),
            embedding_model_name=embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2",
            backend=(backend or env_backend or auto_backend).lower(),
            collection_name=collection_name,
            supabase_url=supabase_url or os.getenv("SUPABASE_URL"),
            supabase_key=supabase_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY"),
            supabase_table=supabase_table or os.getenv("SUPABASE_RAG_TABLE", "rag_documents"),
            supabase_rpc=supabase_rpc or os.getenv("SUPABASE_RAG_RPC", "match_rag_documents"),
            strict=strict,
        )

        self.embedding_model: Optional[SentenceTransformer] = None
        self.initialized = False

        self._chroma_client = None
        self._chroma_collection = None

        self._supabase = None

        logger.info(
            "RAGEngine configurado | backend={} | db_path={}"
            .format(self.config.backend, self.config.database_path)
        )

    def initialize(self):
        if self.initialized:
            return

        if self.config.backend == "none":
            logger.warning("RAG desabilitado (backend=none)")
            self.initialized = True
            return

        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        except Exception as exc:
            if self.config.strict:
                raise
            logger.warning(f"Falha ao carregar embeddings: {exc}. RAG desabilitado.")
            self.config.backend = "none"
            self.initialized = True
            return

        if self.config.backend == "chroma":
            self._initialize_chroma()
        elif self.config.backend == "supabase":
            self._initialize_supabase()
        else:
            if self.config.strict:
                raise ValueError(f"Backend RAG inválido: {self.config.backend}")
            logger.warning(f"Backend RAG inválido '{self.config.backend}'. RAG desabilitado.")
            self.config.backend = "none"

        self.initialized = True

    def _initialize_chroma(self):
        if chromadb is None:
            if self.config.strict:
                raise RuntimeError("chromadb não instalado")
            logger.warning("chromadb não instalado. RAG desabilitado.")
            self.config.backend = "none"
            return

        chroma_path = self.config.database_path / "chromadb"
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self._chroma_collection = self._chroma_client.get_collection(self.config.collection_name)
            logger.info(
                f"Coleção Chroma carregada: {self._chroma_collection.count()} documentos"
            )
        except Exception:
            self._chroma_collection = self._chroma_client.create_collection(
                name=self.config.collection_name,
                metadata={"description": "Scientific papers for geophysics"}
            )
            logger.info("Nova coleção Chroma criada")

    def _initialize_supabase(self):
        if create_client is None:
            if self.config.strict:
                raise RuntimeError("supabase não instalado")
            logger.warning("supabase não instalado. RAG desabilitado.")
            self.config.backend = "none"
            return

        if not self.config.supabase_url or not self.config.supabase_key:
            if self.config.strict:
                raise RuntimeError("SUPABASE_URL/SUPABASE_KEY ausentes")
            logger.warning("SUPABASE_URL/SUPABASE_KEY ausentes. RAG desabilitado.")
            self.config.backend = "none"
            return

        self._supabase = create_client(self.config.supabase_url, self.config.supabase_key)
        logger.info("Supabase conectado para RAG")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.initialized:
            self.initialize()

        if self.config.backend == "none":
            return []

        if self.embedding_model is None:
            return []

        query_embedding = self.embedding_model.encode([query])[0]

        if self.config.backend == "chroma":
            if self._chroma_collection is None or self._chroma_collection.count() == 0:
                return []

            results = self._chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
            return formatted_results

        if self.config.backend == "supabase":
            if self._supabase is None:
                return []

            payload = {
                "query_embedding": query_embedding.tolist(),
                "match_count": top_k
            }
            response = self._supabase.rpc(self.config.supabase_rpc, payload).execute()
            rows = response.data or []

            formatted_results = []
            for row in rows:
                metadata = row.get("metadata")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {"source": "Supabase", "raw": metadata}
                formatted_results.append({
                    "document": row.get("content", ""),
                    "metadata": metadata or {},
                    "distance": row.get("distance")
                })
            return formatted_results

        return []

    def index_documents(
        self,
        force_reindex: bool = False,
        chunk_size: int = 500,
        overlap: int = 50,
        clear_existing: bool = False,
        batch_size: int = 200
    ):
        if not self.initialized:
            self.initialize()

        if self.config.backend == "none":
            logger.warning("RAG desabilitado. Indexação ignorada.")
            return

        pdf_files = list(self.config.database_path.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"Nenhum PDF encontrado em {self.config.database_path}")
            return

        documents, metadatas, ids = [], [], []

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(str(pdf_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                chunks = self._split_text(text, chunk_size=chunk_size, overlap=overlap)

                for i, chunk in enumerate(chunks):
                    chunk_id = self._make_id(pdf_path, i)
                    documents.append(chunk)
                    metadatas.append({
                        "source": pdf_path.name,
                        "path": str(pdf_path),
                        "chunk": i,
                        "total_chunks": len(chunks)
                    })
                    ids.append(chunk_id)
            except Exception as exc:
                logger.error(f"Erro ao processar {pdf_path.name}: {exc}")
                continue

        if not documents:
            logger.warning("Nenhum documento processado")
            return

        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        if self.config.backend == "chroma":
            if force_reindex and self._chroma_client is not None:
                try:
                    self._chroma_client.delete_collection(self.config.collection_name)
                except Exception:
                    pass
                self._initialize_chroma()

            self._chroma_collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.success(f"{len(documents)} chunks indexados no ChromaDB")
            return

        if self.config.backend == "supabase":
            if self._supabase is None:
                logger.warning("Supabase não inicializado")
                return

            if clear_existing:
                self._supabase.table(self.config.supabase_table).delete().neq("id", "").execute()

            rows = []
            for i, doc in enumerate(documents):
                rows.append({
                    "id": ids[i],
                    "content": doc,
                    "metadata": metadatas[i],
                    "embedding": embeddings[i].tolist()
                })

            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                self._supabase.table(self.config.supabase_table).upsert(batch).execute()

            logger.success(f"{len(rows)} chunks indexados no Supabase")

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), max(1, chunk_size - overlap)):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def _make_id(self, pdf_path: Path, chunk_index: int) -> str:
        raw = f"{pdf_path.as_posix()}::{chunk_index}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def format_citation_abnt(self, metadata: Dict[str, Any], text_snippet: str = "") -> str:
        source = metadata.get("source", "Documento desconhecido") if isinstance(metadata, dict) else "Documento desconhecido"

        citation = f"""
> 📚 **Referência:**
> **{source}**
"""

        if text_snippet:
            if len(text_snippet) > 200:
                text_snippet = text_snippet[:200] + "..."
            citation += f"""
> *Trecho relevante:*
> \"{text_snippet}\"
"""

        return citation
