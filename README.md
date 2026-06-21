# RAG-documentQandA (RAG Q&A System)

A Retrieval-Augmented Generation (RAG) question-answering system built to upload documents and query them using an AI assistant.

## Features

- **Multi-format Document Upload**: Supports PDF, TXT, and CSV documents.
- **AI-Powered Q&A**: Ask questions and get answers based entirely on the uploaded context.
- **Source Transparency**: View source documents (chunks) used to generate the answer.
- **Streaming Responses**: Real-time feedback through streaming generation.
- **Evaluation**: Built-in RAGAS evaluation support.

## Technology Stack

- **API Layer**: [FastAPI](https://fastapi.tiangolo.com/)
- **RAG Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM & Embeddings**: Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`) via `langchain-google-genai`
- **Vector Database**: [Qdrant Cloud](https://qdrant.tech/)
- **Document Parsing**: `pypdf`, `python-docx`
- **Containerization**: Docker

## Project Structure

- `app/api/`: FastAPI routes for documents and querying.
- `app/core/`: Core RAG components (`document_processor.py`, `embeddings.py`, `rag_chain.py`, `vector_store.py`, `ragas_evaluator.py`).
- `app/utils/`: Utility functions like logging.
- `app/main.py`: FastAPI application entry point.
- `app/config.py`: Configuration management using `pydantic-settings`.

## Prerequisites

- Python 3.13 (or 3.10+)
- A Google Gemini API Key.
- A Qdrant Cloud URL and API Key (or local Qdrant instance).

## Setup & Configuration

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd RAG-documentQandA
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Linux/macOS use: source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add the following keys based on your configuration:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

## Running with Docker

You can also run the application using the provided Dockerfile:

```bash
docker build -t rag-system .
docker run -p 8000:8000 --env-file .env rag-system
```

## Local Hosting Command

To run the application locally for development, execute the following command from the root of the project:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, you can access the interactive Swagger API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)
