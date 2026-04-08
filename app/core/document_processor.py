"""Document processing module for loading and chunking documents."""

import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """process documents for RAG pipeline"""

    SUPPORTED_EXTENSIONS = {".pdf",".txt", ".csv"}

    def __init__(
            self,
            chunk_size :int | None = None,
            chunk_overlap :int | None = None,
    ):
        """initialize document processor.
        
        Args:
            chunk_size (int | None, optional)
            chunk_overlap (int | None, optional)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        logger.info(
            f"DocumentProcessor initialized with chunk_size = {self.chunk_size}, "
            f"chunk_overlap = {self.chunk_overlap}"
        )
        

    def load_pdf(self, file_path :str | Path) -> list[Document]:
        """
        Load a Pdf
        
        
        Args : 
        file_path : Path to pdf file

        returns: list of Documents
        """

        file_path = Path(file_path)
        logger.info(f"loading PDF : {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        logger.info(f"loaded {len(documents)} pages from PDF : {file_path.name}")
        return documents
    

    def load_text(self, file_path :str | Path) -> list[Document]:
        """load a text file

        Args:
            file_path (str | Path): path to text file

        Returns:
            list[Document]: list of Documents
        """

        file_path = Path(file_path)
        logger.info(f"Loading text file: {file_path.name}")

        loader = TextLoader(str(file_path))
        documents = loader.load()

        logger.info(f"Loaded text file : {file_path.name} with {len(documents)} documents")

        return documents
    
    def load_csv(self, file_path :str | Path ) -> list[Document]:
        """Load a CSV file
        Args : 
            file_path : Path to csv file

        Returns: 
            list[Document]: list of Documents
        
        """

        file_path = Path(file_path)
        logger.info(f"Loading CSV file: {file_path.name}")

        loader = CSVLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        logger.info(f"loaded csv files : {file_path.name} with {len(documents)} documents")

        return documents
    
    def load_file(self, file_path : str | Path) -> list[Document]:
        """ load a file based on its extension
        
        Args: 
            file_path : Path to file 
            
        Return:
            list[Document]: list of Documents

        Raises:
            ValueError : If file extension is not supported
        """

        file_path  = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported extensions are: {', '.join(self.SUPPORTED_EXTENSIONS)}"

            )
        
        loader  = {
            ".pdf" : self.load_pdf,
            ".txt" : self.load_text,
            ".csv" : self.load_csv, 
        }

        return loader[extension](file_path)
    
    def load_from_upload(
        self,
        file: BinaryIO,
        filename: str,
    ) -> list[Document]:
        """Load document from uploaded file.

        Args:
            file: File-like object
            filename: Original filename

        Returns:
            List of Document objects
        """
        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        try:
            documents = self.load_file(tmp_path)

            # Update metadata with original filename
            for doc in documents:
                doc.metadata["source"] = filename

            return documents
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)


    def split_documents(self, documents : list[Document]) -> list[Document]:
        """split documents into chunks
        
        Args: 
            documents : list of document object
            
        Returns : 
            List of chunked document objects
            
            """
        logger.info(f"splitting {len(documents)} documents into chunks")
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"split into {len(chunks)} chunks")

        return chunks

    def process_file(self, file_path : str | Path) -> list[Document]:
        """load and split the file in one step

        Args:
            file_path (str | Path): path to file

        Returns:
            list[Document]: list of chunked document objects
        """

        Document = self.load_file(file_path)
        chunks = self.split_documents(Document)

        return chunks

    def process_upload(
            self,
            file : BinaryIO,
            filename : str,
    ) -> list[Document]:
        """load and split an uploaded file

        Args:
            file (BinaryIO): file-like object
            filename (str): original filename of the uploaded file

        Returns:
            list[Document]: list of chunked document objects
        """

        document = self.load_from_upload(file, filename)
        return self.split_documents(document)
    



        

        

