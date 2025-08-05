#!/usr/bin/env python3
"""
Alternative Google LLM implementation using google-generativeai
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GoogleGenerativeAI:
    """Google Generative AI implementation for LLM calls"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.mock_mode = False
        self._initialize_genai()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Substitute environment variables
            for section in config.values():
                if isinstance(section, dict):
                    for key, value in section.items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            env_var = value[2:-1]
                            section[key] = os.getenv(env_var, value)
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _initialize_genai(self):
        """Initialize Google Generative AI"""
        try:
            # Get API key from environment
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                # Try to get from service account
                service_account_path = self.config.get('google_cloud', {}).get('service_account_path')
                if service_account_path and os.path.exists(service_account_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
                    logger.info("Using service account for authentication")
                else:
                    logger.warning("No Google API key or service account found")
                    self.mock_mode = True
                    return
            
            # Configure the API
            genai.configure(api_key=api_key)
            
            # Initialize the model
            model_name = self.config.get('vertex_ai', {}).get('model_name', 'gemini-pro')
            self.model = genai.GenerativeModel(model_name)
            
            logger.info(f"Google Generative AI initialized with model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Google Generative AI (running in mock mode): {e}")
            self.mock_mode = True
    
    def generate_response(self, query: str, context: str, document_name: Optional[str] = None) -> str:
        """Generate response using Google Generative AI"""
        
        if self.mock_mode:
            return self._generate_mock_response(query, context, document_name)
        
        try:
            # Construct prompt
            prompt = self._construct_prompt(query, context, document_name)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Google Generative AI API call failed: {e}")
            return self._generate_mock_response(query, context, document_name)
    
    def _construct_prompt(self, query: str, context: str, document_name: Optional[str] = None) -> str:
        """Construct prompt for the LLM"""
        
        prompt = f"""You are a helpful financial document analysis assistant. Your task is to answer questions about financial documents based on the provided context.

Question: {query}

Context from the document{f" '{document_name}'" if document_name else ""}:
{context}

Please provide a clear, accurate, and helpful answer based on the context provided. If the information is not available in the context, please state that clearly.

Answer:"""
        
        return prompt
    
    def _generate_mock_response(self, query: str, context: str, document_name: Optional[str] = None) -> str:
        """Generate mock response when AI is not available"""
        
        # Extract key information from context
        context_lines = context.split('\n')
        relevant_info = []
        
        for line in context_lines:
            if any(keyword in line.lower() for keyword in ['balance', 'amount', 'account', 'holder', 'number', 'date']):
                relevant_info.append(line.strip())
        
        mock_response = f"""Based on the context from {document_name or 'the document'}, here's what I found regarding your question: "{query}"

The relevant information from the document includes:
{chr(10).join(relevant_info[:5])}

Note: This is a simulated response. To get AI-generated responses, please:
1. Set up Google API key in environment variables
2. Ensure Google Generative AI API is enabled
3. Restart the application

For now, you can review the source context above to answer your question."""
        
        return mock_response
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of the LLM service"""
        try:
            if self.mock_mode:
                return {
                    "status": "degraded",
                    "message": "Running in mock mode - Google API not configured",
                    "model": "mock"
                }
            
            # Test with a simple prompt
            test_response = self.model.generate_content("Hello")
            return {
                "status": "healthy",
                "message": "Google Generative AI is working",
                "model": self.config.get('vertex_ai', {}).get('model_name', 'unknown')
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Google Generative AI health check failed: {e}",
                "model": "unknown"
            }

def create_llm_client(config_path: str = "config/config.yaml") -> GoogleGenerativeAI:
    """Factory function to create LLM client"""
    return GoogleGenerativeAI(config_path) 