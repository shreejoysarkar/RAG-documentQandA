"""Vector store for qdrant operation."""

from Lib.site-packages.datasets.exceptions import UnexpectedDownloadedFileError
from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpctedResponse
from qdrant_client.http.models import Distance, VectorParams


from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import logger


## Embedding dimension 
EMBEDDING_DIMENSION = 1536

@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get cached qdrant client instance.
    
    Returns:
        QdrantClient: Qdrant client instance.
    """
    logger.info(f"Connecting to Qdrant at :{settings.qdrant_url}")

    client = QdrantClient(
        url = settings.qdrant_url,
        api_key = settings.qdrant_api_key,

    )

    logger.info("Successfully connected to Qdrant.")
    return client


class VectoreStoreService:
    """Serviece for managing vector store operations"""
    
    def __init__(self, collection_name: str | None = None):
        """Initialize vectore store service
        
        Args: 
            collection_name : Name of the qdrant collection
        """

        self.collection_name = collection_name or settings.collection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()

        # making sure collection exists
        self._ensure_collection()

        ## Initialize langchain qdrant vector store
        self.vector_store = QdrantVectoreStore(
            client = self.client,
            collection_name = self.collection_name,
            embeddings = self.embeddings,
        )
        
        logger.info(f"VectoreStoreService initialized with collection: {self.collection_name}")

    def _ensure_collection(self) -> None:
        """Ensure that collection exists, create if not"""
        
        try:
            collection_info = self.client.get_colleciton(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}'exist with"
                f"{collection_info.points_count} points"

                
            )

        except UnexpctedResponse:
            logger.info(f"Creating collection : {self.collection_name}")
            self.client.create_collection(
                collection_name = self.collection_name,
                vectors_config = VectorParams(
                    size = EMBEDDING_DIMENSION,
                    distance = Distance.COSINE,
                )
            )

            logger.info(f"Collection '{self.collection_name}' created successfully")
        

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vectore store.

        Args : 
            documents : List of Document objects to add

        Returns :
            List of document IDs

        """

        if not documents:
            logger.warning("No document to add")
            return []

        logger.info(f"Adding {len(documents)} document to collection")

        ## Generate unique IDs for each document
        ids = [str(uuid4()) for _ in documents]

        # add to vector store 
        self.vector_store.add_documents(documents, ids= ids)

        logger.info(f"Succesfully added {len(documents)} documents")

        return ids

    def search(
        self, query : str, k: int | None = None) -> list[Document]:

        """Search for similar documents.

        Args : 
            query : Search query
            k :Number of results to return  

        Return:
            List of similar document objects
        """
    
        k = k or get_settings.retrieval_k

        logger.debug(f"Searching for : {query[:50]}... (k={k})")

        results = self.vector_store.similarity_search(query, k = k)
        logger.debug(f"found {len(results)} results")
        return results

    
    def search_with_scores(self,query: str,k: int|None = None,) -> list[tuple[Document, float]]:

        """Search for similar documents with relevance scores.

        args: 
            query : Search query
            k : Number of results to return

        Returns:
            List of (document, score) tuples
            """
        k = k or get_settings.retrieval_k

        logger.debug(f"Searching for : {query[:50]}... (k={k})")

        results = self.vector_store.similarity_search_with_score(query, k = k)
        logger.debug(f"found {len(results)} results")
        return results

    def get_retriever(self, k:int | None=None) -> Any:

        """Get a retriever for the vector store.
        Args :
            k : Number of documents to retrieve
        Return :
            Langchain retriever object

            """
        k = k or get_settings.retrieval_k
            
        return self.vectoer_store.as_retriever(
                search_type = "similarity",
                search_kqargs = {'k':k},
            )

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns: 
            Dictionary with collection statistics

        """
        try:
            info = self.client.get_collection(self.collection_name)

            return{
                "name" :self.collection_name,
                "point_count": info.points_count,
                "indexed_vector_count": info.indexed_vectors_count,
                "status" : info.status.value,

            }

        except UnexpctedResponse:
            return {
                "name" : self.collection_name,
                "point_count" : 0,
                "indexed_vector_count" : 0,
                "status" : "not_found",
            }

    def health_check(self) -> bool:

        """Check if vector store is healthy.

        Returns:
            True if healthy, false otherwize"""
        try:

            self.client.get_collection_aliases()
            return True

        except Exception as e:
            logger.error(f"Vector Store health check failed{e}")

            return False
