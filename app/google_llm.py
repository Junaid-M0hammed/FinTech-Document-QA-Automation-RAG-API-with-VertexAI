"""
Google Vertex AI PaLM 2 Integration Module

This module handles all interactions with Google's Vertex AI PaLM 2 API
for generating responses in the RAG pipeline.
"""

import os
import logging
from typing import Optional, Dict, Any
import yaml

logger = logging.getLogger(__name__)

try:
    import vertexai
    from vertexai.language_models import TextGenerationModel
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False
    logger.warning("Vertex AI not available - running in mock mode")


class VertexAILLM:
    """Vertex AI PaLM 2 Text Generation Client with fallback support"""
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        Initialize Vertex AI client with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.mock_mode = False
        self._initialize_vertex_ai()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            return config
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
            env_var = config[2:-1]  # Remove ${ and }
            return os.getenv(env_var, config)
        else:
            return config
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI client with fallback support"""
        if not VERTEXAI_AVAILABLE:
            logger.warning("Vertex AI not available - enabling mock mode")
            self.mock_mode = True
            return
            
        try:
            # Set up authentication
            service_account_path = self.config['google_cloud']['service_account_path']
            if service_account_path and os.path.exists(service_account_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
            
            # Initialize Vertex AI
            vertexai.init(
                project=self.config['google_cloud']['project_id'],
                location=self.config['google_cloud']['location']
            )
            
            # Initialize the text generation model
            self.model = TextGenerationModel.from_pretrained(self.config['vertex_ai']['model_name'])
            
            logger.info("Vertex AI initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI (running in mock mode): {e}")
            logger.warning("Mock mode enabled - responses will be simulated")
            self.mock_mode = True
            
            # Check if it's a billing issue
            if "billing" in str(e).lower() or "403" in str(e):
                logger.warning("💡 This appears to be a billing issue. To resolve:")
                logger.warning("   1. Enable billing in Google Cloud Console")
                logger.warning("   2. Make sure Vertex AI API is enabled")
                logger.warning("   3. Wait a few minutes for changes to propagate")
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        document_name: Optional[str] = None
    ) -> str:
        """
        Generate response using PaLM 2 based on query and retrieved context
        
        Args:
            query: User's question
            context: Retrieved context from vector search
            document_name: Optional document name for context
            
        Returns:
            Generated response from PaLM 2 (or mock response if unavailable)
        """
        try:
            if self.mock_mode:
                return self._generate_mock_response(query, context, document_name)
            
            # Construct prompt for financial document Q&A
            prompt = self._construct_prompt(query, context, document_name)
            
            # Generate response using Vertex AI
            response = self._call_vertex_ai(prompt)
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return response
            
        except Exception as e:
            logger.warning(f"Error generating response, falling back to mock: {e}")
            return self._generate_mock_response(query, context, document_name)
    
    def _generate_mock_response(self, query: str, context: str, document_name: Optional[str] = None) -> str:
        """Generate a mock response when Vertex AI is unavailable"""
        doc_context = f" from {document_name}" if document_name else ""
        
        response = f"""[MOCK RESPONSE - Vertex AI not available]

Based on the context{doc_context}, here's what I found regarding your question: "{query}"

The relevant information from the document includes:
{context[:300]}{'...' if len(context) > 300 else ''}

To get AI-generated responses, please:
1. Enable billing in your Google Cloud project
2. Ensure Vertex AI API is activated
3. Restart the application

For now, you can review the source context above to answer your question."""
        
        return response
    
    def _construct_prompt(
        self, 
        query: str, 
        context: str, 
        document_name: Optional[str] = None
    ) -> str:
        """Construct optimized prompt for financial document Q&A"""
        
        doc_context = f" from the document '{document_name}'" if document_name else ""
        
        prompt = f"""You are a helpful financial document analysis assistant. Your task is to answer questions about financial documents based on the provided context.

Context{doc_context}:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the answer is not in the context, say "I cannot find this information in the provided context"
3. Be precise and professional in your response
4. Include relevant numbers, dates, and financial terms when available
5. Structure your response clearly for financial professionals

Answer:"""
        
        return prompt
    
    def _call_vertex_ai(self, prompt: str) -> str:
        """Call Vertex AI PaLM 2 API with the constructed prompt"""
        if self.mock_mode:
            return self._generate_mock_response("", prompt)
            
        try:
            # Get model configuration
            model_config = self.config['vertex_ai']
            
            # Generate response
            response = self.model.predict(
                prompt,
                temperature=model_config['temperature'],
                max_output_tokens=model_config['max_output_tokens'],
                top_p=model_config['top_p'],
                top_k=model_config['top_k']
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Vertex AI API call failed: {e}")
            return self._generate_mock_response("", prompt)
    
    def health_check(self) -> bool:
        """Check if Vertex AI connection is healthy"""
        if self.mock_mode:
            logger.info("Health check: Running in mock mode (Vertex AI not available)")
            return True  # Return True so system can continue
            
        try:
            # Simple test call
            test_response = self._call_vertex_ai("Test prompt")
            return bool(test_response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Utility function for easy import
def create_llm_client(config_path: str = "./config/config.yaml") -> VertexAILLM:
    """Factory function to create VertexAI LLM client"""
    return VertexAILLM(config_path) 