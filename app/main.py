"""
Fintech Document Q&A API

Production-grade FastAPI application for financial document intelligence
using RAG architecture with Pinecone and Google Vertex AI PaLM 2.
"""

import os
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .rag_pipeline import create_rag_pipeline, RAGPipeline
from .google_llm_alt import create_llm_client, GoogleGenerativeAI

# Load environment variables at startup
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
rag_pipeline: Optional[RAGPipeline] = None
llm_client: Optional[GoogleGenerativeAI] = None
app_config: Optional[Dict[str, Any]] = None

# Security
security = HTTPBearer()


class UploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    chunks_created: Optional[int] = Field(None, description="Number of text chunks created")


class QueryRequest(BaseModel):
    """Request model for document queries"""
    question: str = Field(..., description="User's question about the document(s)")
    document_id: Optional[str] = Field(None, description="Optional filter by specific document ID")
    top_k: Optional[int] = Field(5, description="Number of relevant chunks to retrieve")


class QueryResponse(BaseModel):
    """Response model for document queries"""
    answer: str = Field(..., description="Generated answer based on document context")
    question: str = Field(..., description="Original question")
    document_id: Optional[str] = Field(None, description="Document ID if filtered")
    sources: List[Dict[str, Any]] = Field(..., description="Source chunks used for the answer")
    confidence_scores: List[float] = Field(..., description="Relevance scores for each source")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, bool] = Field(..., description="Health status of individual components")
    version: str = Field(..., description="API version")


def load_config(config_path: str = "./config/config.yaml") -> Dict[str, Any]:
    """Load application configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        def substitute_env_vars(obj):
            if isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        return substitute_env_vars(config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    expected_api_key = app_config['api']['api_key']
    
    if not expected_api_key or expected_api_key.startswith('${'):
        # API key not configured properly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured"
        )
    
    if credentials.credentials != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_pipeline, llm_client, app_config
    
    try:
        # Load configuration
        app_config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = create_rag_pipeline()
        logger.info("RAG pipeline initialized")
        
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = create_llm_client()
        logger.info("LLM client initialized")
        
        # Create necessary directories
        upload_dir = Path(app_config['storage']['upload_dir'])
        upload_dir.mkdir(exist_ok=True)
        
        temp_dir = Path(app_config['storage']['temp_dir'])
        temp_dir.mkdir(exist_ok=True)
        
        logs_dir = Path("./logs")
        logs_dir.mkdir(exist_ok=True)
        
        logger.info("Fintech RAG API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down Fintech RAG API")


# Initialize FastAPI app
app = FastAPI(
    title="Fintech Document Q&A API",
    description="Production-grade RAG API for financial document intelligence using Google PaLM 2 and Pinecone",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fintech Document Q&A API",
        "version": "1.0.0",
        "description": "Production-grade RAG API for financial document intelligence",
        "endpoints": {
            "upload": "/upload - Upload financial documents (PDF/CSV/TXT)",
            "ask": "/ask - Ask questions about uploaded documents",
            "health": "/health - System health check"
        }
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (PDF, CSV, or TXT)"),
    api_key: str = Depends(verify_api_key)
) -> UploadResponse:
    """
    Upload and process a financial document for Q&A
    
    Supported formats:
    - PDF: Bank statements, contracts, reports
    - CSV: Financial data, transaction records
    - TXT: Plain text financial documents
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower().replace('.', '')
        supported_formats = app_config['document_processing']['supported_formats']
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported: {', '.join(supported_formats)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document through RAG pipeline
            document_id = await rag_pipeline.process_document(temp_file_path, file.filename)
            
            logger.info(f"Successfully processed document: {file.filename} -> {document_id}")
            
            return UploadResponse(
                document_id=document_id,
                filename=file.filename,
                status="success",
                message="Document uploaded and processed successfully",
                chunks_created=None  # Could be enhanced to return actual count
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@app.post("/ask", response_model=QueryResponse, tags=["Q&A"])
async def ask_question(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> QueryResponse:
    """
    Ask questions about uploaded financial documents
    
    Examples:
    - "What is the monthly fee for this account?"
    - "Summarize the key terms and conditions"
    - "What was the highest transaction amount in March?"
    - "What are the interest rates mentioned?"
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Retrieve relevant context from vector database
        context, source_chunks = await rag_pipeline.query_documents(
            query=request.question,
            document_id=request.document_id,
            top_k=request.top_k
        )
        
        if not context:
            return QueryResponse(
                answer="I couldn't find relevant information to answer your question. Please make sure you've uploaded documents and try rephrasing your question.",
                question=request.question,
                document_id=request.document_id,
                sources=[],
                confidence_scores=[]
            )
        
        # Generate response using LLM
        document_name = None
        if source_chunks and source_chunks[0]['metadata'].get('filename'):
            document_name = source_chunks[0]['metadata']['filename']
        
        answer = llm_client.generate_response(
            query=request.question,
            context=context,
            document_name=document_name
        )
        
        # Prepare response with sources
        sources = []
        confidence_scores = []
        
        for chunk in source_chunks:
            sources.append({
                "text": chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                "metadata": {
                    "filename": chunk['metadata'].get('filename'),
                    "chunk_index": chunk['metadata'].get('chunk_index'),
                    "document_id": chunk['metadata'].get('document_id')
                }
            })
            confidence_scores.append(round(chunk['score'], 3))
        
        logger.info(f"Successfully answered question: {request.question[:50]}...")
        
        return QueryResponse(
            answer=answer,
            question=request.question,
            document_id=request.document_id,
            sources=sources,
            confidence_scores=confidence_scores
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Check the health status of all system components
    """
    try:
        # Check RAG pipeline components
        rag_health = rag_pipeline.health_check()
        
        # Check LLM client
        llm_health = llm_client.health_check()
        
        # Combine health status
        components = {
            **rag_health,
            "llm_client": llm_health
        }
        
        # Determine overall status
        overall_status = "healthy" if all(components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            components=components,
            version=app_config['api']['version']
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            components={"error": False},
            version="1.0.0"
        )


@app.get("/documents/{document_id}/info", tags=["Documents"])
async def get_document_info(
    document_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get information about a specific document"""
    try:
        # This would require additional implementation in the RAG pipeline
        # to store and retrieve document metadata
        return {
            "document_id": document_id,
            "message": "Document info endpoint - implementation pending"
        }
    except Exception as e:
        logger.error(f"Document info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=False,  # Set to True for development
        log_level="info"
    ) 