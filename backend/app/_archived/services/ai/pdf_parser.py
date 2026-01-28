"""
PDF Parser for RAG System
Extracts text, metadata and creates chunks from scientific PDFs
"""

from typing import List, Dict, Any, Optional
import logging
import re
from pathlib import Path
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parser for scientific PDF documents with intelligent chunking
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Initialize PDF parser
        
        Args:
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
            min_chunk_size: Minimum chunk size to keep
        """
        if PyPDF2 is None:
            logger.warning("PyPDF2 not installed. PDF parsing will not work.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF file and extract text, metadata, and chunks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with:
                - text: Full extracted text
                - metadata: Document metadata
                - chunks: List of text chunks with metadata
                - pages: Number of pages
        """
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")
        
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Parsing PDF: {path.name}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = self._extract_metadata(reader, path)
                
                # Extract text from all pages
                text = ""
                page_texts = []
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_texts.append((page_num, page_text))
                            text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                
                # Clean text
                text = self._clean_text(text)
                
                # Create chunks
                chunks = self._create_chunks(text, metadata, page_texts)
                
                result = {
                    "text": text,
                    "metadata": metadata,
                    "chunks": chunks,
                    "pages": len(reader.pages),
                    "file_size": path.stat().st_size,
                    "parsed_at": datetime.utcnow().isoformat()
                }
                
                logger.info(
                    f"Parsed PDF: {path.name} - "
                    f"{len(reader.pages)} pages, "
                    f"{len(text)} chars, "
                    f"{len(chunks)} chunks"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise
    
    def _extract_metadata(self, reader: PyPDF2.PdfReader, path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF
        
        Args:
            reader: PyPDF2 reader object
            path: Path to PDF file
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "filename": path.name,
            "file_path": str(path),
        }
        
        # Try to extract PDF metadata
        if reader.metadata:
            try:
                if reader.metadata.title:
                    metadata["title"] = reader.metadata.title
                if reader.metadata.author:
                    metadata["author"] = reader.metadata.author
                if reader.metadata.subject:
                    metadata["subject"] = reader.metadata.subject
                if reader.metadata.creator:
                    metadata["creator"] = reader.metadata.creator
                if reader.metadata.producer:
                    metadata["producer"] = reader.metadata.producer
                if reader.metadata.creation_date:
                    metadata["creation_date"] = str(reader.metadata.creation_date)
            except Exception as e:
                logger.warning(f"Error extracting PDF metadata: {e}")
        
        # Extract title from filename if not in metadata
        if "title" not in metadata:
            # Remove extension and clean filename
            title = path.stem
            title = re.sub(r'[-_]', ' ', title)
            title = re.sub(r'\s+', ' ', title).strip()
            metadata["title"] = title
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        
        # Remove page headers/footers patterns (common in academic papers)
        # This is a simple heuristic and can be improved
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) > 3 or not line.isdigit():
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def _create_chunks(
        self,
        text: str,
        metadata: Dict[str, Any],
        page_texts: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        Create text chunks with intelligent splitting
        
        Args:
            text: Full document text
            metadata: Document metadata
            page_texts: List of (page_num, text) tuples
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Strategy 1: Try to split by sections (for scientific papers)
        sections = self._split_by_sections(text)
        
        if len(sections) > 1:
            # Process each section
            for section_idx, section in enumerate(sections):
                section_chunks = self._split_text_recursive(
                    section,
                    self.chunk_size,
                    self.chunk_overlap
                )
                
                for chunk_idx, chunk_text in enumerate(section_chunks):
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk = {
                            "text": chunk_text,
                            "metadata": {
                                **metadata,
                                "chunk_index": len(chunks),
                                "section_index": section_idx,
                                "chunk_in_section": chunk_idx,
                                "total_chunks": None  # Will be set later
                            }
                        }
                        chunks.append(chunk)
        else:
            # No clear sections, use page-based chunking
            for page_num, page_text in page_texts:
                page_chunks = self._split_text_recursive(
                    page_text,
                    self.chunk_size,
                    self.chunk_overlap
                )
                
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk = {
                            "text": chunk_text,
                            "metadata": {
                                **metadata,
                                "chunk_index": len(chunks),
                                "page_number": page_num,
                                "chunk_in_page": chunk_idx,
                                "total_chunks": None
                            }
                        }
                        chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """
        Try to split text by sections (for scientific papers)
        
        Args:
            text: Document text
            
        Returns:
            List of section texts
        """
        # Common section headers in scientific papers
        section_patterns = [
            r'\n\s*(?:ABSTRACT|Abstract)\s*\n',
            r'\n\s*(?:INTRODUCTION|Introduction)\s*\n',
            r'\n\s*(?:METHODS|Methods|METHODOLOGY|Methodology)\s*\n',
            r'\n\s*(?:RESULTS|Results)\s*\n',
            r'\n\s*(?:DISCUSSION|Discussion)\s*\n',
            r'\n\s*(?:CONCLUSION|Conclusion|CONCLUSIONS|Conclusions)\s*\n',
            r'\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*\n',
            r'\n\s*\d+\.\s+[A-Z][^.!?\n]{5,50}\s*\n',  # Numbered sections
        ]
        
        # Find all section markers
        markers = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                markers.append(match.start())
        
        if not markers:
            return [text]
        
        # Sort markers and remove duplicates
        markers = sorted(set(markers))
        
        # Split text by markers
        sections = []
        for i in range(len(markers)):
            start = markers[i]
            end = markers[i + 1] if i + 1 < len(markers) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        
        # If first marker is not at start, include text before first marker
        if markers[0] > 0:
            intro = text[:markers[0]].strip()
            if intro:
                sections.insert(0, intro)
        
        return sections if len(sections) > 1 else [text]
    
    def _split_text_recursive(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recursively split text into chunks
        
        Args:
            text: Text to split
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            separators: List of separators to try (in order)
            
        Returns:
            List of text chunks
        """
        if separators is None:
            # Try to split on natural boundaries
            separators = [
                '\n\n',  # Paragraphs
                '\n',    # Lines
                '. ',    # Sentences
                ' ',     # Words
                ''       # Characters (last resort)
            ]
        
        if len(text) <= chunk_size:
            return [text] if text else []
        
        # Try each separator
        for separator in separators:
            if separator == '':
                # Character-level split (last resort)
                chunks = []
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    if chunk:
                        chunks.append(chunk)
                return chunks
            
            # Split by separator
            splits = text.split(separator)
            
            chunks = []
            current_chunk = ""
            
            for split in splits:
                # Add separator back (except for empty splits)
                if split:
                    test_chunk = current_chunk + (separator if current_chunk else '') + split
                    
                    if len(test_chunk) <= chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Current chunk is full
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # Start new chunk
                        if len(split) > chunk_size:
                            # Split is too large, needs recursive splitting
                            sub_chunks = self._split_text_recursive(
                                split,
                                chunk_size,
                                overlap,
                                separators[separators.index(separator) + 1:]
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # If we got good chunks, return them
            if chunks and all(len(c) <= chunk_size * 1.5 for c in chunks):
                return chunks
        
        # Fallback: just split at chunk_size
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    
    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract citations from text (simple heuristic)
        
        Args:
            text: Document text
            
        Returns:
            List of potential citations
        """
        citations = []
        
        # Look for patterns like "Author et al. (2020)"
        pattern1 = r'([A-Z][a-z]+(?:\s+et\s+al\.)?\s*\(\d{4}\))'
        matches = re.findall(pattern1, text)
        citations.extend([{"citation": m, "type": "inline"} for m in matches])
        
        # Look for DOI links
        pattern2 = r'10\.\d{4,}/[^\s]+'
        matches = re.findall(pattern2, text)
        citations.extend([{"doi": m, "type": "doi"} for m in matches])
        
        return citations


# Global instance
_parser_instance = None


def get_parser() -> PDFParser:
    """Get global PDF parser instance"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = PDFParser()
    return _parser_instance
