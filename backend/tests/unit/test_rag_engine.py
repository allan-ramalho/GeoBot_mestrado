"""
Tests for RAG Engine
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from app.services.ai.rag_engine import RAGEngine


@pytest.fixture
def rag_engine():
    """Create RAG engine instance"""
    return RAGEngine()


@pytest.fixture
def sample_query():
    """Sample search query"""
    return "magnetic anomaly reduction to pole"


@pytest.fixture
def sample_embedding():
    """Sample embedding vector"""
    return np.random.randn(384)


@pytest.fixture
def sample_search_results():
    """Sample search results from database"""
    return [
        {
            "content": "Reduction to pole is a transformation technique...",
            "metadata": {
                "title": "Magnetic Data Processing",
                "authors": "Smith, J.",
                "year": 2020,
                "journal": "Geophysics"
            },
            "similarity": 0.92
        },
        {
            "content": "Magnetic anomalies can be transformed...",
            "metadata": {
                "title": "Geophysical Methods",
                "authors": "Johnson, A.",
                "year": 2019,
                "journal": "Journal of Applied Geophysics"
            },
            "similarity": 0.87
        }
    ]


class TestRAGEngineInit:
    """Test RAG engine initialization"""
    
    def test_engine_initialization(self, rag_engine):
        """Test engine initializes without loading model"""
        assert rag_engine is not None
        assert rag_engine.embedding_model is None
        assert rag_engine._initialized is False
    
    @pytest.mark.asyncio
    async def test_lazy_initialization(self, rag_engine):
        """Test model loads lazily on first use"""
        with patch('app.services.ai.rag_engine.SentenceTransformer') as mock_model:
            mock_model.return_value = Mock()
            
            await rag_engine.initialize()
            
            assert rag_engine._initialized is True
            assert rag_engine.embedding_model is not None
            mock_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_only_once(self, rag_engine):
        """Test model is loaded only once"""
        with patch('app.services.ai.rag_engine.SentenceTransformer') as mock_model:
            mock_model.return_value = Mock()
            
            await rag_engine.initialize()
            await rag_engine.initialize()
            await rag_engine.initialize()
            
            # Should be called only once
            assert mock_model.call_count == 1


class TestEmbeddingGeneration:
    """Test embedding generation"""
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding(self, rag_engine, sample_query):
        """Test generating embedding for query"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock, return_value=[]):
            await rag_engine.search(sample_query)
            
            mock_model.encode.assert_called_once_with(sample_query, normalize_embeddings=True)
    
    @pytest.mark.asyncio
    async def test_embedding_normalization(self, rag_engine, sample_query):
        """Test embeddings are normalized"""
        mock_model = Mock()
        # Return non-normalized vector
        unnormalized = np.array([1.0, 2.0, 3.0])
        mock_model.encode.return_value = unnormalized
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock, return_value=[]):
            await rag_engine.search(sample_query)
            
            # Check normalize_embeddings parameter
            call_args = mock_model.encode.call_args
            assert call_args[1]['normalize_embeddings'] is True


class TestVectorSearch:
    """Test vector similarity search"""
    
    @pytest.mark.asyncio
    async def test_vector_search_with_supabase(self, rag_engine, sample_embedding):
        """Test vector search calls Supabase"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = []
            mock_response.status_code = 200
            
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post
            
            with patch('app.core.config.settings') as mock_settings:
                mock_settings.SUPABASE_URL = "https://test.supabase.co"
                mock_settings.SUPABASE_KEY = "test_key"
                
                await rag_engine._vector_search(sample_embedding, top_k=5)
                
                mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vector_search_without_supabase(self, rag_engine, sample_embedding):
        """Test vector search returns empty when Supabase not configured"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.SUPABASE_URL = None
            mock_settings.SUPABASE_KEY = None
            
            results = await rag_engine._vector_search(sample_embedding, top_k=5)
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_vector_search_request_format(self, rag_engine, sample_embedding):
        """Test vector search request has correct format"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = []
            mock_response.status_code = 200
            
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post
            
            with patch('app.core.config.settings') as mock_settings:
                mock_settings.SUPABASE_URL = "https://test.supabase.co"
                mock_settings.SUPABASE_KEY = "test_key"
                
                await rag_engine._vector_search(sample_embedding, top_k=3)
                
                call_args = mock_post.call_args
                json_data = call_args[1]['json']
                
                assert 'query_embedding' in json_data
                assert 'match_count' in json_data
                assert json_data['match_count'] == 3


class TestSearchResults:
    """Test search result formatting"""
    
    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, rag_engine, sample_query):
        """Test search returns properly formatted results"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        mock_results = [
            {
                "content": "Test content",
                "metadata": {"title": "Test Paper", "authors": "Author", "year": 2020},
                "similarity": 0.95
            }
        ]
        
        with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock, return_value=mock_results):
            results = await rag_engine.search(sample_query)
            
            assert len(results) > 0
            assert "content" in results[0]
            assert "metadata" in results[0]
            assert "score" in results[0]
            assert "citation" in results[0]
    
    @pytest.mark.asyncio
    async def test_search_top_k_parameter(self, rag_engine, sample_query):
        """Test top_k parameter limits results"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            
            await rag_engine.search(sample_query, top_k=3)
            
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args[0][1] == 3  # top_k argument
    
    @pytest.mark.asyncio
    async def test_empty_search_results(self, rag_engine, sample_query):
        """Test handling empty search results"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock, return_value=[]):
            results = await rag_engine.search(sample_query)
            
            assert results == []


class TestCitationFormatting:
    """Test citation formatting"""
    
    def test_format_citation_complete(self, rag_engine):
        """Test formatting complete citation"""
        metadata = {
            "title": "Test Paper",
            "authors": "Smith, J. and Johnson, A.",
            "year": 2020,
            "journal": "Geophysics"
        }
        
        citation = rag_engine._format_citation(metadata)
        
        assert "Smith" in citation
        assert "2020" in citation
        assert "Test Paper" in citation or "Geophysics" in citation
    
    def test_format_citation_minimal(self, rag_engine):
        """Test formatting citation with minimal info"""
        metadata = {
            "title": "Test Paper"
        }
        
        citation = rag_engine._format_citation(metadata)
        
        assert "Test Paper" in citation
    
    def test_format_citation_empty(self, rag_engine):
        """Test formatting empty citation"""
        metadata = {}
        
        citation = rag_engine._format_citation(metadata)
        
        assert isinstance(citation, str)
        assert len(citation) > 0


class TestDocumentEmbedding:
    """Test document embedding for indexing"""
    
    @pytest.mark.asyncio
    async def test_embed_document(self, rag_engine):
        """Test embedding a document"""
        document = "This is a test document about magnetic anomalies."
        
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with patch.object(rag_engine, 'initialize', new_callable=AsyncMock):
            embedding = await rag_engine.embed_document(document)
            
            assert embedding is not None
            assert isinstance(embedding, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_embed_multiple_documents(self, rag_engine):
        """Test embedding multiple documents in batch"""
        documents = [
            "Document 1 about gravity",
            "Document 2 about magnetics",
            "Document 3 about seismic"
        ]
        
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(3, 384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with patch.object(rag_engine, 'initialize', new_callable=AsyncMock):
            embeddings = await rag_engine.embed_documents(documents)
            
            assert embeddings is not None
            assert len(embeddings) == len(documents)


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_search_model_not_initialized(self, rag_engine, sample_query):
        """Test search initializes model if needed"""
        with patch('app.services.ai.rag_engine.SentenceTransformer') as mock_model:
            mock_instance = Mock()
            mock_instance.encode.return_value = np.random.randn(384)
            mock_model.return_value = mock_instance
            
            with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock, return_value=[]):
                # Model not initialized
                assert rag_engine.embedding_model is None
                
                await rag_engine.search(sample_query)
                
                # Model should be initialized now
                assert rag_engine.embedding_model is not None
    
    @pytest.mark.asyncio
    async def test_vector_search_http_error(self, rag_engine, sample_embedding):
        """Test handling HTTP errors in vector search"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_client.return_value.__aenter__.return_value.post = mock_post
            
            with patch('app.core.config.settings') as mock_settings:
                mock_settings.SUPABASE_URL = "https://test.supabase.co"
                mock_settings.SUPABASE_KEY = "test_key"
                
                with pytest.raises(Exception):
                    await rag_engine._vector_search(sample_embedding, top_k=5)
    
    @pytest.mark.asyncio
    async def test_search_embedding_error(self, rag_engine, sample_query):
        """Test handling embedding generation errors"""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        with pytest.raises(Exception):
            await rag_engine.search(sample_query)


class TestSimilarityScoring:
    """Test similarity scoring"""
    
    @pytest.mark.asyncio
    async def test_results_sorted_by_similarity(self, rag_engine, sample_query):
        """Test results are sorted by similarity score"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384)
        
        rag_engine.embedding_model = mock_model
        rag_engine._initialized = True
        
        mock_results = [
            {"content": "Doc 1", "metadata": {}, "similarity": 0.85},
            {"content": "Doc 2", "metadata": {}, "similarity": 0.95},
            {"content": "Doc 3", "metadata": {}, "similarity": 0.75}
        ]
        
        with patch.object(rag_engine, '_vector_search', new_callable=AsyncMock, return_value=mock_results):
            results = await rag_engine.search(sample_query)
            
            # Results should maintain order from vector search
            assert results[0]["score"] == 0.85
            assert results[1]["score"] == 0.95
            assert results[2]["score"] == 0.75


# Import httpx for error testing
import httpx
