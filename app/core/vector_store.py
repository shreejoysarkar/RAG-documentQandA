"""Vector store for qdrant operation."""

from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
