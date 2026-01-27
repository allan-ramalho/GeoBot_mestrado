#!/usr/bin/env python
"""
Script para ingestão de PDFs no sistema RAG
Baixa PDFs do Supabase, processa e gera embeddings
"""

import asyncio
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import tempfile
import os

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.ai.rag_engine import RAGEngine
from app.services.ai.pdf_parser import PDFParser
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def download_pdfs_from_supabase(rag: RAGEngine) -> List[Dict[str, Any]]:
    """
    Download PDF files from Supabase storage
    
    Returns:
        List of PDF file information
    """
    try:
        # List files in PDF bucket
        response = rag.supabase.storage.from_(settings.SUPABASE_PDF_BUCKET).list()
        
        if not response:
            logger.warning("No files found in Supabase storage")
            return []
        
        pdf_files = []
        temp_dir = Path(tempfile.gettempdir()) / "geobot_pdfs"
        temp_dir.mkdir(exist_ok=True)
        
        for file_info in response:
            if file_info['name'].lower().endswith('.pdf'):
                logger.info(f"📥 Downloading: {file_info['name']}")
                
                try:
                    # Download file
                    file_data = rag.supabase.storage.from_(settings.SUPABASE_PDF_BUCKET).download(file_info['name'])
                    
                    # Save to temp file
                    temp_file = temp_dir / file_info['name']
                    with open(temp_file, 'wb') as f:
                        f.write(file_data)
                    
                    pdf_files.append({
                        'name': file_info['name'],
                        'path': str(temp_file),
                        'size': file_info.get('metadata', {}).get('size', 0),
                        'updated_at': file_info.get('updated_at')
                    })
                    
                    logger.info(f"✅ Downloaded: {file_info['name']}")
                    
                except Exception as e:
                    logger.error(f"❌ Error downloading {file_info['name']}: {e}")
        
        return pdf_files
        
    except Exception as e:
        logger.error(f"Error accessing Supabase storage: {e}")
        return []


async def process_pdf(
    pdf_path: str,
    pdf_parser: PDFParser,
    rag: RAGEngine
) -> Dict[str, Any]:
    """
    Process single PDF: parse, chunk, and generate embeddings
    
    Args:
        pdf_path: Path to PDF file
        pdf_parser: PDFParser instance
        rag: RAGEngine instance
        
    Returns:
        Processing result with statistics
    """
    logger.info(f"🔄 Processing: {Path(pdf_path).name}")
    
    try:
        # Parse PDF
        parsed = pdf_parser.parse_pdf(pdf_path)
        
        logger.info(
            f"📄 Parsed {parsed['pages']} pages, "
            f"{len(parsed['chunks'])} chunks"
        )
        
        # Generate embeddings for each chunk
        chunks_with_embeddings = []
        
        for i, chunk in enumerate(parsed['chunks']):
            if i % 10 == 0:
                logger.info(f"  Embedding chunk {i+1}/{len(parsed['chunks'])}")
            
            # Generate embedding
            embedding = await rag._generate_embedding(chunk['text'])
            
            chunks_with_embeddings.append({
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'embedding': embedding
            })
        
        # Store in Supabase
        logger.info(f"💾 Storing {len(chunks_with_embeddings)} chunks in database...")
        
        stored_count = 0
        for chunk_data in chunks_with_embeddings:
            try:
                # Insert into documents table
                result = rag.supabase.table('documents').insert({
                    'content': chunk_data['text'],
                    'embedding': chunk_data['embedding'],
                    'metadata': chunk_data['metadata']
                }).execute()
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing chunk: {e}")
        
        logger.info(f"✅ Stored {stored_count}/{len(chunks_with_embeddings)} chunks")
        
        return {
            'success': True,
            'filename': parsed['metadata']['filename'],
            'pages': parsed['pages'],
            'chunks': len(chunks_with_embeddings),
            'stored': stored_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}")
        return {
            'success': False,
            'filename': Path(pdf_path).name,
            'error': str(e)
        }


async def main():
    """Main ingestion workflow"""
    
    print("=" * 60)
    print("📚 GeoBot PDF Ingestion System")
    print("=" * 60)
    
    # Check configuration
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        print("\n❌ Error: Supabase not configured!")
        print("Please set SUPABASE_URL and SUPABASE_KEY in backend/.env")
        return
    
    print(f"\n🔧 Configuration:")
    print(f"  Supabase URL: {settings.SUPABASE_URL}")
    print(f"  Bucket: {settings.SUPABASE_PDF_BUCKET}")
    print(f"  Embedding Model: {settings.RAG_EMBEDDING_MODEL}")
    print(f"  Chunk Size: {settings.RAG_CHUNK_SIZE}")
    
    # Initialize services
    print("\n🚀 Initializing services...")
    rag = RAGEngine()
    await rag.initialize()
    print("✅ RAG engine initialized")
    
    pdf_parser = PDFParser(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP
    )
    print("✅ PDF parser initialized")
    
    # Download PDFs from Supabase
    print("\n📥 Downloading PDFs from Supabase...")
    pdf_files = await download_pdfs_from_supabase(rag)
    
    if not pdf_files:
        print("\n⚠️  No PDFs found in Supabase storage")
        print("\nTo add PDFs:")
        print("1. Go to Supabase Dashboard")
        print("2. Navigate to Storage")
        print(f"3. Upload PDFs to '{settings.SUPABASE_PDF_BUCKET}' bucket")
        print("4. Run this script again")
        return
    
    print(f"✅ Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    print("\n🔄 Processing PDFs...")
    print("-" * 60)
    
    results = []
    for pdf_file in pdf_files:
        result = await process_pdf(
            pdf_file['path'],
            pdf_parser,
            rag
        )
        results.append(result)
        print("-" * 60)
    
    # Summary
    print("\n📊 Ingestion Summary:")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        total_pages = sum(r['pages'] for r in successful)
        total_chunks = sum(r['chunks'] for r in successful)
        total_stored = sum(r['stored'] for r in successful)
        
        print(f"\n📈 Statistics:")
        print(f"  Total pages processed: {total_pages}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Total chunks stored: {total_stored}")
    
    if failed:
        print(f"\n⚠️  Failed files:")
        for r in failed:
            print(f"  - {r['filename']}: {r['error']}")
    
    print("\n✅ Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
