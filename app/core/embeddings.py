"""Embedding generation using gemini embedding API."""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

@lru_cache
def get_embeddings() -> GeminiEmbeddings:
    """Get an instance of the GeminiEmbeddings class."""
    
    settings = get_settings()
    logger.info("initializing GeminiEmbeddings")

    embeddings = GoogleGenerativeAIEmbeddings(
        model = settings.embedding_model,
        gemini_api_key = settings.GEMINI_API_KEY,
    )

    logger.info("GeminiEmbeddings initialized successfully")
    return embeddings


class EmbeddingService:
    """Service for generating embeddings using Gemini API."""

    def __init__(self):
        """Initialize embedding services"""
        
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.model_name = settings.embedding_model

    def embed_query(self, text:str) -> list[float]:
        """Generate embeddings for a single query
        
        Args:
            text : query text
            
        Returns:
            embedding vector for the query as list"""
        
        logger.debug(f"Generating embedding for query : {text[:50]}....")
        return self.embeddings.embed_query(text)
    

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents"""
        
        logger.debug(f"Generating embeddings for {len(texts)} documents")
        return self.embeddings.embed_documents(texts)
    



