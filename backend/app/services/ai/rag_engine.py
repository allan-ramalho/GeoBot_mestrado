"""
RAG Engine
Retrieval Augmented Generation for scientific literature
Uses Supabase + pgvector for semantic search
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

import httpx
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine for scientific literature retrieval
    """
    
    def __init__(self):
        self.embedding_model = None
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize embedding model
        Lazy loading to avoid startup delay
        """
        if not self._initialized:
            logger.info(f"📚 Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._initialized = True
            logger.info("✅ RAG Engine initialized")
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with metadata and scores
        """
        try:
            await self.initialize()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            
            # Search in Supabase using pgvector
            results = await self._vector_search(query_embedding, top_k)
            
            # Format results with citations
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": result["similarity"],
                    "citation": self._format_citation(result["metadata"])
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ RAG search failed: {e}")
            raise
    
    async def _vector_search(
        self,
        embedding: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Supabase
        """
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("⚠️ Supabase not configured, returning empty results")
            return []
        
        try:
            # Call Supabase RPC function for vector search
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.SUPABASE_URL}/rest/v1/rpc/match_documents",
                    headers={
                        "apikey": settings.SUPABASE_KEY,
                        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "query_embedding": embedding.tolist(),
                        "match_count": top_k,
                        "match_threshold": 0.5
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Supabase search failed: {response.text}")
                    return []
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _format_citation(self, metadata: Dict[str, Any]) -> str:
        """
        Format citation from metadata
        """
        authors = metadata.get("authors", "Unknown")
        year = metadata.get("year", "n.d.")
        title = metadata.get("title", "Untitled")
        journal = metadata.get("journal", "")
        
        citation = f"{authors} ({year}). {title}"
        if journal:
            citation += f". {journal}"
        
        return citation
    
    async def ingest_documents(self, documents: List[Dict[str, Any]]):
        """
        Ingest documents into RAG system
        
        Args:
            documents: List of documents with content and metadata
        """
        try:
            await self.initialize()
            
            logger.info(f"📥 Ingesting {len(documents)} documents...")
            
            for doc in documents:
                # Generate embeddings
                content = doc["content"]
                chunks = self._chunk_text(content)
                
                embeddings = self.embedding_model.encode(
                    chunks,
                    normalize_embeddings=True,
                    show_progress_bar=True
                )
                
                # Store in Supabase
                await self._store_embeddings(
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=doc["metadata"]
                )
            
            logger.info("✅ Document ingestion completed")
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding
        """
        words = text.split()
        chunks = []
        
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def _store_embeddings(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        metadata: Dict[str, Any]
    ):
        """
        Store embeddings in Supabase
        """
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("⚠️ Supabase not configured, skipping storage")
            return
        
        try:
            async with httpx.AsyncClient() as client:
                for chunk, embedding in zip(chunks, embeddings):
                    await client.post(
                        f"{settings.SUPABASE_URL}/rest/v1/documents",
                        headers={
                            "apikey": settings.SUPABASE_KEY,
                            "Authorization": f"Bearer {settings.SUPABASE_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "content": chunk,
                            "embedding": embedding.tolist(),
                            "metadata": metadata
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    async def download_pdfs_from_supabase(self) -> List[Path]:
        """
        Download PDFs from Supabase storage bucket
        """
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("⚠️ Supabase not configured")
            return []
        
        try:
            # List files in bucket
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.SUPABASE_URL}/storage/v1/object/list/{settings.SUPABASE_BUCKET}",
                    headers={
                        "apikey": settings.SUPABASE_KEY,
                        "Authorization": f"Bearer {settings.SUPABASE_KEY}"
                    }
                )
                
                files = response.json()
                
                # Download each PDF
                pdf_dir = settings.DATA_DIR / "pdfs"
                pdf_dir.mkdir(parents=True, exist_ok=True)
                
                downloaded_files = []
                for file_info in files:
                    if file_info["name"].endswith(".pdf"):
                        file_path = await self._download_pdf(file_info["name"], pdf_dir)
                        downloaded_files.append(file_path)
                
                logger.info(f"📥 Downloaded {len(downloaded_files)} PDFs")
                return downloaded_files
                
        except Exception as e:
            logger.error(f"Failed to download PDFs: {e}")
            return []
    
    async def _download_pdf(self, filename: str, output_dir: Path) -> Path:
        """
        Download single PDF from Supabase
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/storage/v1/object/public/{settings.SUPABASE_BUCKET}/{filename}",
                headers={
                    "apikey": settings.SUPABASE_KEY
                }
            )
            
            file_path = output_dir / filename
            file_path.write_bytes(response.content)
            
            return file_path
