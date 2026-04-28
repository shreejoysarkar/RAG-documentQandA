""" RAG chain module using Langchian LCEL"""

from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


from app.config import get_settings
from app.core.vector_store import VectoreStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

## Rag prompt template

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context: {context}

Question: {question}

Answer:"""

def format_docs(docs: list[Document]) -> str:
    """Format documents into a single context string.
    Args :
        docs : List of document objects

    Returns:
        Formatted context string
    """

    return "\n\n---\n\n".join(doc.page_content for doc in docs)

class RagChain:
    """Rag Chain for question answering"""
    def _init__(self, vector_store_service: VectoreStoreService | None = None):
        """Initilize RAG Chain.
        ARgs:
            vector_store_service : Optional VectorStoreService instance
            """
        self.vector_store =vector_store_service or VectoreStoreService()
        self.retriever = self.vector_store.get_retriever()

        ## Initialize evaluator
        self._evaluator = None

        ## Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model = settings.llm_model,
            temperature = settings.llm_temperature,
            google_api_key = settings.google_api_key
        )

        ## Initialize prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        ## Build LCEL chain
        self.chain = (
            {
                "context" : self.retriever | format_docs,
                "question" : RunnablePassthrough(),
                
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info(
            f"RAG Chain Initialized"
        )

    @property
    def evaluator(self):
        