"""
Comprehensive test suite for Fintech RAG API

Tests cover all major functionality including:
- Document upload and processing
- Question answering
- Authentication
- Error handling
- Health checks
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import yaml

# Mock the dependencies before importing the app
with patch('app.rag_pipeline.create_rag_pipeline'), \
     patch('app.google_llm.create_llm_client'), \
     patch('app.main.load_config') as mock_load_config:
    
    # Mock configuration
    mock_config = {
        'api': {
            'api_key': 'test-api-key',
            'title': 'Test API',
            'description': 'Test Description',
            'version': '1.0.0',
            'host': '0.0.0.0',
            'port': 8000
        },
        'document_processing': {
            'supported_formats': ['pdf', 'csv', 'txt'],
            'max_file_size_mb': 10
        },
        'storage': {
            'upload_dir': './test_uploads',
            'temp_dir': './test_temp'
        }
    }
    mock_load_config.return_value = mock_config
    
    from app.main import app

# Test configuration
TEST_API_KEY = "test-api-key"
INVALID_API_KEY = "invalid-api-key"

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def auth_headers():
    """Authentication headers for testing"""
    return {"Authorization": f"Bearer {TEST_API_KEY}"}

@pytest.fixture
def invalid_auth_headers():
    """Invalid authentication headers for testing"""
    return {"Authorization": f"Bearer {INVALID_API_KEY}"}

@pytest.fixture
def sample_text_file():
    """Create sample text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample financial document.\nAccount balance: $1,000.00\nInterest rate: 2.5%")
        return f.name

@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("date,amount,description\n2024-01-01,1000.00,Deposit\n2024-01-02,-50.00,ATM Withdrawal")
        return f.name

@pytest.fixture
def sample_large_file():
    """Create large file for testing file size limits"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Create a file larger than 10MB
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        f.write(large_content)
        return f.name

class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

class TestAuthentication:
    """Test API authentication"""
    
    def test_upload_without_auth(self, client, sample_text_file):
        """Test upload endpoint without authentication"""
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        assert response.status_code == 403  # Missing auth header
    
    def test_upload_with_invalid_auth(self, client, sample_text_file, invalid_auth_headers):
        """Test upload endpoint with invalid authentication"""
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")},
                headers=invalid_auth_headers
            )
        assert response.status_code == 401
    
    def test_ask_without_auth(self, client):
        """Test ask endpoint without authentication"""
        response = client.post(
            "/ask",
            json={"question": "What is the balance?"}
        )
        assert response.status_code == 403
    
    def test_ask_with_invalid_auth(self, client, invalid_auth_headers):
        """Test ask endpoint with invalid authentication"""
        response = client.post(
            "/ask",
            json={"question": "What is the balance?"},
            headers=invalid_auth_headers
        )
        assert response.status_code == 401

class TestDocumentUpload:
    """Test document upload functionality"""
    
    @patch('app.main.rag_pipeline')
    def test_upload_text_file(self, mock_rag_pipeline, client, sample_text_file, auth_headers):
        """Test uploading a text file"""
        mock_rag_pipeline.process_document = AsyncMock(return_value="test-doc-id")
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "document_id" in data
        assert data["filename"] == "test.txt"
    
    @patch('app.main.rag_pipeline')
    def test_upload_csv_file(self, mock_rag_pipeline, client, sample_csv_file, auth_headers):
        """Test uploading a CSV file"""
        mock_rag_pipeline.process_document = AsyncMock(return_value="test-doc-id")
        
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.csv", f, "text/csv")},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_upload_unsupported_format(self, client, auth_headers):
        """Test uploading unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.exe') as f:
            f.write(b"fake executable")
            f.seek(0)
            response = client.post(
                "/upload",
                files={"file": ("test.exe", f, "application/octet-stream")},
                headers=auth_headers
            )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    def test_upload_without_filename(self, client, auth_headers):
        """Test uploading file without filename"""
        response = client.post(
            "/upload",
            files={"file": ("", b"content", "text/plain")},
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "Filename is required" in response.json()["detail"]
    
    @patch('app.main.rag_pipeline')
    def test_upload_processing_error(self, mock_rag_pipeline, client, sample_text_file, auth_headers):
        """Test upload when document processing fails"""
        mock_rag_pipeline.process_document = AsyncMock(side_effect=Exception("Processing failed"))
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")},
                headers=auth_headers
            )
        
        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

class TestQuestionAnswering:
    """Test question answering functionality"""
    
    @patch('app.main.rag_pipeline')
    @patch('app.main.llm_client')
    def test_ask_question(self, mock_llm_client, mock_rag_pipeline, client, auth_headers):
        """Test asking a question"""
        # Mock RAG pipeline response
        mock_rag_pipeline.query_documents = AsyncMock(return_value=(
            "Sample context about balance",
            [{"text": "Balance: $1000", "score": 0.95, "metadata": {"filename": "test.txt"}}]
        ))
        
        # Mock LLM response
        mock_llm_client.generate_response.return_value = "The balance is $1000."
        
        response = client.post(
            "/ask",
            json={"question": "What is the balance?"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["question"] == "What is the balance?"
        assert len(data["sources"]) > 0
        assert len(data["confidence_scores"]) > 0
    
    @patch('app.main.rag_pipeline')
    def test_ask_question_no_context(self, mock_rag_pipeline, client, auth_headers):
        """Test asking question when no relevant context is found"""
        mock_rag_pipeline.query_documents = AsyncMock(return_value=("", []))
        
        response = client.post(
            "/ask",
            json={"question": "What is the balance?"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "couldn't find relevant information" in data["answer"]
    
    def test_ask_empty_question(self, client, auth_headers):
        """Test asking empty question"""
        response = client.post(
            "/ask",
            json={"question": ""},
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "Question cannot be empty" in response.json()["detail"]
    
    def test_ask_question_missing_field(self, client, auth_headers):
        """Test asking question without required field"""
        response = client.post(
            "/ask",
            json={},
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.main.rag_pipeline')
    @patch('app.main.llm_client')
    def test_ask_with_document_id(self, mock_llm_client, mock_rag_pipeline, client, auth_headers):
        """Test asking question with specific document ID"""
        mock_rag_pipeline.query_documents = AsyncMock(return_value=(
            "Sample context",
            [{"text": "Answer text", "score": 0.9, "metadata": {"filename": "test.txt"}}]
        ))
        mock_llm_client.generate_response.return_value = "Answer"
        
        response = client.post(
            "/ask",
            json={
                "question": "What is the balance?",
                "document_id": "test-doc-id",
                "top_k": 3
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test-doc-id"
    
    @patch('app.main.rag_pipeline')
    def test_ask_processing_error(self, mock_rag_pipeline, client, auth_headers):
        """Test question processing error"""
        mock_rag_pipeline.query_documents = AsyncMock(side_effect=Exception("Query failed"))
        
        response = client.post(
            "/ask",
            json={"question": "What is the balance?"},
            headers=auth_headers
        )
        
        assert response.status_code == 500
        assert "Failed to process question" in response.json()["detail"]

class TestHealthCheck:
    """Test health check functionality"""
    
    @patch('app.main.rag_pipeline')
    @patch('app.main.llm_client')
    def test_health_check_healthy(self, mock_llm_client, mock_rag_pipeline, client):
        """Test health check when all systems are healthy"""
        mock_rag_pipeline.health_check.return_value = {
            "pinecone": True,
            "embedding_model": True
        }
        mock_llm_client.health_check.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["pinecone"] is True
        assert data["components"]["llm_client"] is True
    
    @patch('app.main.rag_pipeline')
    @patch('app.main.llm_client')
    def test_health_check_degraded(self, mock_llm_client, mock_rag_pipeline, client):
        """Test health check when some systems are unhealthy"""
        mock_rag_pipeline.health_check.return_value = {
            "pinecone": False,  # Unhealthy
            "embedding_model": True
        }
        mock_llm_client.health_check.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["pinecone"] is False
    
    @patch('app.main.rag_pipeline')
    def test_health_check_error(self, mock_rag_pipeline, client):
        """Test health check when error occurs"""
        mock_rag_pipeline.health_check.side_effect = Exception("Health check failed")
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"

class TestDocumentInfo:
    """Test document info endpoint"""
    
    def test_get_document_info(self, client, auth_headers):
        """Test getting document info"""
        response = client.get("/documents/test-doc-id/info", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test-doc-id"
    
    def test_get_document_info_without_auth(self, client):
        """Test getting document info without authentication"""
        response = client.get("/documents/test-doc-id/info")
        
        assert response.status_code == 403

class TestErrorHandling:
    """Test error handling"""
    
    def test_http_exception_handler(self, client, auth_headers):
        """Test HTTP exception handling"""
        # Trigger a 404 error
        response = client.get("/nonexistent-endpoint", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"] is True

class TestAsyncEndpoints:
    """Test async endpoint functionality"""
    
    @pytest.mark.asyncio
    @patch('app.main.rag_pipeline')
    async def test_async_upload(self, mock_rag_pipeline, async_client, sample_text_file, auth_headers):
        """Test async upload functionality"""
        mock_rag_pipeline.process_document = AsyncMock(return_value="test-doc-id")
        
        with open(sample_text_file, 'rb') as f:
            response = await async_client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")},
                headers=auth_headers
            )
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    @patch('app.main.rag_pipeline')
    @patch('app.main.llm_client')
    async def test_async_ask(self, mock_llm_client, mock_rag_pipeline, async_client, auth_headers):
        """Test async ask functionality"""
        mock_rag_pipeline.query_documents = AsyncMock(return_value=(
            "Context",
            [{"text": "Answer", "score": 0.9, "metadata": {"filename": "test.txt"}}]
        ))
        mock_llm_client.generate_response.return_value = "Response"
        
        response = await async_client.post(
            "/ask",
            json={"question": "Test question?"},
            headers=auth_headers
        )
        
        assert response.status_code == 200

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_files(sample_text_file, sample_csv_file, sample_large_file):
    """Cleanup test files after each test"""
    yield
    for file_path in [sample_text_file, sample_csv_file, sample_large_file]:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 