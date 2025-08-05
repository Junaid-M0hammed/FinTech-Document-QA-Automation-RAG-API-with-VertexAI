"""
RAG Pipeline Module

This module handles the complete RAG pipeline including:
- Document parsing (PDF, CSV, TXT)
- Text chunking and preprocessing
- Vector embedding generation
- Pinecone vector database operations
- Context retrieval for Q&A
"""

import os
import uuid
import logging
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
import fitz  # PyMuPDF
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import aiofiles

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles parsing of different document types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = config['document_processing']['supported_formats']
        self.max_file_size = config['document_processing']['max_file_size_mb'] * 1024 * 1024
    
    async def process_document(self, file_path: str, filename: str) -> List[str]:
        """
        Process uploaded document and extract text content
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            List of text chunks extracted from the document
        """
        # Validate file
        if not await self._validate_file(file_path, filename):
            raise ValueError(f"File validation failed for {filename}")
        
        # Extract text based on file type
        file_extension = Path(filename).suffix.lower().replace('.', '')
        
        if file_extension == 'pdf':
            text_content = await self._extract_pdf_text(file_path)
        elif file_extension == 'csv':
            text_content = await self._extract_csv_text(file_path)
        elif file_extension == 'txt':
            text_content = await self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if not text_content.strip():
            raise ValueError("No text content could be extracted from the document")
        
        logger.info(f"Extracted {len(text_content)} characters from {filename}")
        return text_content
    
    async def _validate_file(self, file_path: str, filename: str) -> bool:
        """Validate file size and format"""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.error(f"File too large: {file_size} bytes")
                return False
            
            # Check file extension
            file_extension = Path(filename).suffix.lower().replace('.', '')
            if file_extension not in self.supported_formats:
                logger.error(f"Unsupported format: {file_extension}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text_content = ""
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            return text_content
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise
    
    async def _extract_csv_text(self, file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to structured text
            text_content = f"CSV Document with {len(df)} rows and {len(df.columns)} columns\n\n"
            text_content += f"Columns: {', '.join(df.columns)}\n\n"
            
            # Add column descriptions if available
            text_content += "Data Summary:\n"
            for col in df.columns:
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        text_content += f"{col}: {df[col].describe().to_string()}\n\n"
                    else:
                        unique_vals = df[col].unique()[:10]  # First 10 unique values
                        text_content += f"{col}: Sample values - {', '.join(map(str, unique_vals))}\n"
                except Exception:
                    continue
            
            # Add sample rows
            text_content += f"\nSample Data (first 5 rows):\n{df.head().to_string()}\n"
            
            return text_content
        except Exception as e:
            logger.error(f"CSV extraction error: {e}")
            raise
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                async with aiofiles.open(file_path, 'r', encoding='latin-1') as file:
                    content = await file.read()
                return content
            except Exception as e:
                logger.error(f"TXT extraction error: {e}")
                raise
        except Exception as e:
            logger.error(f"TXT extraction error: {e}")
            raise


class TextChunker:
    """Handles text chunking for optimal embedding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config['document_processing']['chunk_size']
        self.chunk_overlap = config['document_processing']['chunk_overlap']
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, document_id: str, filename: str) -> List[Document]:
        """
        Split text into chunks and create LangChain Document objects
        
        Args:
            text: Raw text content
            document_id: Unique document identifier
            filename: Original filename
            
        Returns:
            List of LangChain Document objects
        """
        try:
            # Split text into chunks
            chunks = self.splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "document_id": document_id,
                            "filename": filename,
                            "chunk_index": i,
                            "chunk_id": f"{document_id}_chunk_{i}",
                            "source": filename
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks for document {filename}")
            return documents
        except Exception as e:
            logger.error(f"Text chunking error: {e}")
            raise


class VectorStore:
    """Handles Pinecone vector database operations using Pinecone v7 API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = SentenceTransformer(config['rag']['embedding_model'])
        self.dimension = config['pinecone']['dimension']
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index using v7 API"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.config['pinecone']['api_key'])
            
            index_name = self.config['pinecone']['index_name']
            
            # List existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            # Create index if it doesn't exist
            if index_name not in existing_indexes:
                self.pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            # Get index reference
            self.index = self.pc.Index(index_name)
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Pinecone initialization error: {e}")
            raise
    
    def embed_documents(self, documents: List[Document]) -> List[Tuple[str, List[float], Dict]]:
        """
        Generate embeddings for document chunks
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of tuples (chunk_id, embedding, metadata)
        """
        try:
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # Prepare data for Pinecone
            vectors = []
            for doc, embedding in zip(documents, embeddings):
                vectors.append((
                    doc.metadata['chunk_id'],
                    embedding.tolist(),
                    {
                        **doc.metadata,
                        'text': doc.page_content
                    }
                ))
            
            logger.info(f"Generated embeddings for {len(vectors)} chunks")
            return vectors
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    def store_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        """Store vectors in Pinecone using v7 API"""
        try:
            # Prepare vectors for upsert (new API format)
            upsert_data = []
            for vector_id, embedding, metadata in vectors:
                upsert_data.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors)} vectors in Pinecone")
        except Exception as e:
            logger.error(f"Vector storage error: {e}")
            raise
    
    def search_similar(self, query: str, document_id: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks based on query using v7 API
        
        Args:
            query: User query
            document_id: Optional filter by document ID
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Prepare filter
            filter_dict = {}
            if document_id:
                filter_dict['document_id'] = document_id
            
            # Search Pinecone using new API
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Process results
            results = []
            for match in search_results['matches']:
                results.append({
                    'score': match['score'],
                    'text': match['metadata']['text'],
                    'metadata': match['metadata']
                })
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            raise


class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize RAG pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.document_processor = DocumentProcessor(self.config)
        self.text_chunker = TextChunker(self.config)
        self.vector_store = VectorStore(self.config)
        
        # Ensure upload directory exists
        upload_dir = Path(self.config['storage']['upload_dir'])
        upload_dir.mkdir(exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and process configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            return self._substitute_env_vars(config)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    async def process_document(self, file_path: str, filename: str) -> str:
        """
        Process and store a document in the vector database
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            Document ID for future reference
        """
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Process document
            text_content = await self.document_processor.process_document(file_path, filename)
            
            # Chunk text
            documents = self.text_chunker.chunk_text(text_content, document_id, filename)
            
            # Generate embeddings
            vectors = self.vector_store.embed_documents(documents)
            
            # Store in vector database
            self.vector_store.store_vectors(vectors)
            
            logger.info(f"Successfully processed document {filename} with ID {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def query_documents(
        self, 
        query: str, 
        document_id: Optional[str] = None, 
        top_k: Optional[int] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Query documents and retrieve relevant context
        
        Args:
            query: User's question
            document_id: Optional filter by specific document
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (combined_context, individual_chunks)
        """
        try:
            if top_k is None:
                top_k = self.config['rag']['top_k_retrieval']
            
            # Search for similar chunks
            similar_chunks = self.vector_store.search_similar(query, document_id, top_k)
            
            # Filter by similarity threshold
            threshold = self.config['rag']['similarity_threshold']
            filtered_chunks = [
                chunk for chunk in similar_chunks 
                if chunk['score'] >= threshold
            ]
            
            if not filtered_chunks:
                logger.warning(f"No chunks found above similarity threshold {threshold}")
                return "", []
            
            # Combine context
            combined_context = "\n\n".join([
                f"[Relevance: {chunk['score']:.3f}] {chunk['text']}"
                for chunk in filtered_chunks
            ])
            
            logger.info(f"Retrieved {len(filtered_chunks)} relevant chunks for query")
            return combined_context, filtered_chunks
        except Exception as e:
            logger.error(f"Document query failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components"""
        health_status = {}
        
        try:
            # Check Pinecone connection
            index_stats = self.vector_store.index.describe_index_stats()
            health_status['pinecone'] = True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            health_status['pinecone'] = False
        
        try:
            # Check embedding model
            test_embedding = self.vector_store.embedding_model.encode(["test"])
            health_status['embedding_model'] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding model health check failed: {e}")
            health_status['embedding_model'] = False
        
        return health_status


# Factory function for easy import
def create_rag_pipeline(config_path: str = "./config/config.yaml") -> RAGPipeline:
    """Factory function to create RAG pipeline"""
    return RAGPipeline(config_path) 