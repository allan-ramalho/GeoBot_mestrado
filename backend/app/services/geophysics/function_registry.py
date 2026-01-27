"""
Function Registry
Auto-discovers and registers geophysical processing functions
Enables semantic search and natural language function calling
"""

from typing import List, Dict, Any, Callable, Optional
import inspect
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """
    Registry for geophysical processing functions
    Supports semantic search for function discovery
    """
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.embedding_model = None
        self.function_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False
    
    def register_function(
        self,
        name: str,
        func: Callable,
        description: str,
        keywords: List[str],
        parameters: Dict[str, Any],
        examples: Optional[List[str]] = None
    ):
        """
        Register a processing function
        
        Args:
            name: Function name
            func: Callable function
            description: Detailed description
            keywords: Keywords for semantic search
            parameters: Expected parameters with types
            examples: Usage examples
        """
        self.functions[name] = {
            "name": name,
            "function": func,
            "description": description,
            "keywords": keywords,
            "parameters": parameters,
            "examples": examples or [],
            "docstring": inspect.getdoc(func)
        }
        
        logger.debug(f"✅ Registered function: {name}")
    
    async def initialize(self):
        """Initialize embedding model for semantic search"""
        if not self._initialized:
            logger.info("🔧 Initializing function registry...")
            from app.core.config import settings
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            # Generate embeddings for all functions
            for name, info in self.functions.items():
                # Create searchable text from description + keywords + docstring
                searchable_text = f"{info['description']} {' '.join(info['keywords'])} {info['docstring']}"
                embedding = self.embedding_model.encode(searchable_text, normalize_embeddings=True)
                self.function_embeddings[name] = embedding
            
            self._initialized = True
            logger.info(f"✅ Function registry initialized with {len(self.functions)} functions")
    
    def list_functions(self) -> List[Dict[str, Any]]:
        """List all registered functions"""
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "keywords": info["keywords"],
                "parameters": info["parameters"]
            }
            for info in self.functions.values()
        ]
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a function"""
        info = self.functions.get(name)
        if info:
            return {
                "name": info["name"],
                "description": info["description"],
                "keywords": info["keywords"],
                "parameters": info["parameters"],
                "examples": info["examples"],
                "docstring": info["docstring"]
            }
        return None
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get function by name"""
        info = self.functions.get(name)
        return info["function"] if info else None
    
    async def search_functions(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for functions based on query
        
        Args:
            query: Natural language query
            top_k: Number of results
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of matching functions with similarity scores
        """
        await self.initialize()
        
        if not self.functions:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        
        # Calculate similarities
        similarities = []
        for name, func_embedding in self.function_embeddings.items():
            similarity = np.dot(query_embedding, func_embedding)
            if similarity >= threshold:
                similarities.append((name, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for name, score in similarities[:top_k]:
            info = self.functions[name]
            results.append({
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
                "similarity_score": float(score)
            })
        
        return results
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """
        Get function schemas for OpenAI function calling format
        """
        schemas = []
        
        for info in self.functions.values():
            schema = {
                "name": info["name"],
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": info["parameters"],
                    "required": list(info["parameters"].keys())
                }
            }
            schemas.append(schema)
        
        return schemas


# Global registry instance
_registry = FunctionRegistry()


def register(
    name: str,
    description: str,
    keywords: List[str],
    parameters: Dict[str, Any],
    examples: Optional[List[str]] = None
):
    """
    Decorator for registering functions
    
    Usage:
        @register(
            name="reduction_to_pole",
            description="Apply reduction to pole transformation",
            keywords=["RTP", "reduction", "pole", "magnetic"],
            parameters={"inclination": {"type": "number"}, "declination": {"type": "number"}}
        )
        def reduction_to_pole(data, inclination, declination):
            ...
    """
    def decorator(func: Callable) -> Callable:
        _registry.register_function(
            name=name,
            func=func,
            description=description,
            keywords=keywords,
            parameters=parameters,
            examples=examples
        )
        return func
    
    return decorator


def get_registry() -> FunctionRegistry:
    """Get global registry instance"""
    return _registry
