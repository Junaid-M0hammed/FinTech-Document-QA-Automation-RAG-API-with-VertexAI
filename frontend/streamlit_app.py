#!/usr/bin/env python3
"""
Perfect Streamlit Frontend for Fintech RAG API
Advanced dashboard with real-time analytics, document visualization, and interactive features
"""

import streamlit as st
import requests
import json
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
from datetime import datetime, timedelta
import base64
import io
import hashlib
from collections import defaultdict
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Fintech Document Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "fintech-rag-secure-key-2024"

# Custom CSS for perfect styling
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    /* Cards */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Response Box */
    .response-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    
    /* Source Items */
    .source-item {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .source-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.3);
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        border-color: #007bff;
        background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);
    }
    
    /* Analytics Cards */
    .analytics-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Chat Interface */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .chat-message {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .chat-user {
        background: #007bff;
        color: white;
        margin-left: 20%;
    }
    .chat-ai {
        background: #f8f9fa;
        margin-right: 20%;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
    
    /* File Upload */
    .stFileUploader > div > div > div > div {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedFintechRAGInterface:
    """Advanced Streamlit interface for Fintech document Q&A"""
    
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {API_KEY}"}
        self.uploaded_docs = {}
        self.query_history = []
        self.chat_history = []
        self.analytics_data = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_confidence': 0.0,
            'documents_processed': 0,
            'query_times': [],
            'hourly_usage': defaultdict(int),
            'document_types': defaultdict(int),
            'query_categories': defaultdict(int)
        }
        self.session_start = datetime.now()
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"Cannot connect to API: {str(e)}"}
    
    def upload_document(self, uploaded_file) -> Dict[str, Any]:
        """Upload document to RAG API"""
        if uploaded_file is None:
            return {"status": "error", "message": "Please select a document to upload"}
        
        try:
            start_time = time.time()
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Upload to API
            with open(tmp_path, 'rb') as f:
                files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                response = requests.post(
                    f"{API_BASE_URL}/upload",
                    files=files,
                    headers=self.headers,
                    timeout=30
                )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if response.status_code == 200:
                result = response.json()
                doc_id = result["document_id"]
                filename = result["filename"]
                
                # Store document info
                self.uploaded_docs[doc_id] = {
                    'filename': filename,
                    'upload_time': datetime.now(),
                    'file_size': len(uploaded_file.getvalue()),
                    'file_type': uploaded_file.type or 'unknown',
                    'chunks_created': result.get('chunks_created', 0)
                }
                
                # Update analytics
                self.analytics_data['documents_processed'] += 1
                self.analytics_data['document_types'][uploaded_file.type or 'unknown'] += 1
                
                return {
                    "status": "success",
                    "document_id": doc_id,
                    "filename": filename,
                    "message": "Document uploaded successfully!",
                    "processing_time": time.time() - start_time,
                    "chunks_created": result.get('chunks_created', 0)
                }
            else:
                error_data = response.json()
                return {"status": "error", "message": f"Upload failed: {error_data.get('message', 'Unknown error')}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Upload error: {str(e)}"}
    
    def ask_question(self, question: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Ask question about uploaded documents"""
        if not question.strip():
            return {"status": "error", "message": "Please enter a question"}
        
        try:
            start_time = time.time()
            
            # Prepare request
            payload = {
                "question": question,
                "top_k": 3
            }
            if document_id:
                payload["document_id"] = document_id
            
            # Send question to API
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Update analytics
                self.analytics_data['total_queries'] += 1
                self.analytics_data['successful_queries'] += 1
                self.analytics_data['query_times'].append(processing_time)
                self.analytics_data['hourly_usage'][datetime.now().hour] += 1
                
                # Categorize query
                query_category = self._categorize_query(question)
                self.analytics_data['query_categories'][query_category] += 1
                
                # Calculate average confidence
                if result.get('confidence_scores'):
                    avg_confidence = sum(result['confidence_scores']) / len(result['confidence_scores'])
                    self.analytics_data['avg_confidence'] = (
                        (self.analytics_data['avg_confidence'] * (self.analytics_data['successful_queries'] - 1) + avg_confidence) 
                        / self.analytics_data['successful_queries']
                    )
                
                # Store query history
                self.query_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'timestamp': datetime.now(),
                    'processing_time': processing_time,
                    'confidence_scores': result.get('confidence_scores', []),
                    'document_id': document_id,
                    'category': query_category
                })
                
                # Add to chat history
                self.chat_history.append({
                    'type': 'user',
                    'content': question,
                    'timestamp': datetime.now()
                })
                self.chat_history.append({
                    'type': 'ai',
                    'content': result['answer'],
                    'timestamp': datetime.now(),
                    'sources': result.get('sources', []),
                    'confidence': result.get('confidence_scores', [])
                })
                
                return {
                    "status": "success",
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "confidence_scores": result["confidence_scores"],
                    "processing_time": processing_time,
                    "category": query_category
                }
            else:
                error_data = response.json()
                self.analytics_data['total_queries'] += 1
                return {"status": "error", "message": f"Query failed: {error_data.get('message', 'Unknown error')}"}
                
        except requests.exceptions.RequestException as e:
            self.analytics_data['total_queries'] += 1
            return {"status": "error", "message": f"Connection error: {str(e)}"}
        except Exception as e:
            self.analytics_data['total_queries'] += 1
            return {"status": "error", "message": f"Query error: {str(e)}"}
    
    def _categorize_query(self, question: str) -> str:
        """Categorize the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['balance', 'amount', 'total']):
            return 'Balance Inquiry'
        elif any(word in question_lower for word in ['transaction', 'payment', 'deposit', 'withdrawal']):
            return 'Transaction History'
        elif any(word in question_lower for word in ['fee', 'charge', 'cost']):
            return 'Fees & Charges'
        elif any(word in question_lower for word in ['holder', 'account', 'customer']):
            return 'Account Information'
        elif any(word in question_lower for word in ['limit', 'credit', 'loan']):
            return 'Credit & Limits'
        elif any(word in question_lower for word in ['summary', 'overview', 'report']):
            return 'Summary & Reports'
        else:
            return 'General Inquiry'

def create_advanced_analytics_charts(interface: AdvancedFintechRAGInterface):
    """Create advanced interactive analytics charts"""
    if not interface.query_history:
        return None, None, None, None, None
    
    # Query performance over time
    df_queries = pd.DataFrame(interface.query_history)
    df_queries['hour'] = df_queries['timestamp'].dt.hour
    df_queries['date'] = df_queries['timestamp'].dt.date
    
    # Performance chart
    fig_performance = px.line(
        df_queries, 
        x='timestamp', 
        y='processing_time',
        title='Query Processing Time Over Time',
        labels={'processing_time': 'Processing Time (seconds)', 'timestamp': 'Time'},
        color_discrete_sequence=['#007bff']
    )
    fig_performance.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Confidence distribution
    all_confidences = []
    for query in interface.query_history:
        all_confidences.extend(query.get('confidence_scores', []))
    
    if all_confidences:
        fig_confidence = px.histogram(
            x=all_confidences,
            title='Confidence Score Distribution',
            labels={'x': 'Confidence Score', 'y': 'Frequency'},
            nbins=20,
            color_discrete_sequence=['#28a745']
        )
        fig_confidence.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    else:
        fig_confidence = None
    
    # Query volume by hour
    hourly_queries = df_queries.groupby('hour').size().reset_index(name='count')
    fig_volume = px.bar(
        hourly_queries,
        x='hour',
        y='count',
        title='Query Volume by Hour',
        labels={'hour': 'Hour of Day', 'count': 'Number of Queries'},
        color_discrete_sequence=['#ffc107']
    )
    fig_volume.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Query categories
    if interface.analytics_data['query_categories']:
        categories_df = pd.DataFrame([
            {'Category': k, 'Count': v} 
            for k, v in interface.analytics_data['query_categories'].items()
        ])
        fig_categories = px.pie(
            categories_df,
            values='Count',
            names='Category',
            title='Query Categories Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_categories.update_layout(height=300)
    else:
        fig_categories = None
    
    # Document types
    if interface.analytics_data['document_types']:
        doc_types_df = pd.DataFrame([
            {'Type': k, 'Count': v} 
            for k, v in interface.analytics_data['document_types'].items()
        ])
        fig_doc_types = px.bar(
            doc_types_df,
            x='Type',
            y='Count',
            title='Document Types Processed',
            color_discrete_sequence=['#17a2b8']
        )
        fig_doc_types.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    else:
        fig_doc_types = None
    
    return fig_performance, fig_confidence, fig_volume, fig_categories, fig_doc_types

def main():
    """Main Streamlit application"""
    
    # Initialize interface
    if 'interface' not in st.session_state:
        st.session_state.interface = AdvancedFintechRAGInterface()
    
    interface = st.session_state.interface
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Fintech Document Intelligence Platform</h1>
        <p>Advanced AI-Powered Financial Document Analysis | RAG + Vertex AI</p>
        <p><em>Enterprise-grade document processing with intelligent Q&A and real-time analytics</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for system status and navigation
    with st.sidebar:
        st.markdown("## System Status")
        
        # Health check
        health_result = interface.check_api_health()
        if health_result["status"] == "healthy":
            st.markdown('<span class="status-indicator status-online"></span>API Online', unsafe_allow_html=True)
            st.success("System Ready")
        else:
            st.markdown('<span class="status-indicator status-offline"></span>API Offline', unsafe_allow_html=True)
            st.error("System Issues")
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigation",
            ["Document Q&A", "Chat Interface", "Analytics Dashboard", "System Health", "Document Management", "Settings"]
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("## Quick Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(interface.uploaded_docs))
            st.metric("Total Queries", interface.analytics_data['total_queries'])
        with col2:
            success_rate = interface.analytics_data['successful_queries']/max(interface.analytics_data['total_queries'], 1)*100 if interface.analytics_data['total_queries'] > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Avg Confidence", f"{interface.analytics_data['avg_confidence']:.1%}" if interface.analytics_data['avg_confidence'] > 0 else "0%")
        
        # Session info
        st.markdown("---")
        st.markdown("## Session Info")
        session_duration = datetime.now() - interface.session_start
        st.write(f"**Session Duration:** {str(session_duration).split('.')[0]}")
        st.write(f"**Documents Processed:** {interface.analytics_data['documents_processed']}")
        st.write(f"**Queries Made:** {interface.analytics_data['total_queries']}")
    
    # Main content based on navigation
    if page == "Document Q&A":
        render_advanced_qa_interface(interface)
    elif page == "Chat Interface":
        render_chat_interface(interface)
    elif page == "Analytics Dashboard":
        render_advanced_analytics_dashboard(interface)
    elif page == "System Health":
        render_advanced_system_health(interface)
    elif page == "Document Management":
        render_advanced_document_management(interface)
    elif page == "Settings":
        render_settings(interface)

def render_advanced_qa_interface(interface: AdvancedFintechRAGInterface):
    """Render the advanced Q&A interface"""
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## Document Upload")
        
        # File uploader with enhanced styling
        uploaded_file = st.file_uploader(
            "Choose a financial document",
            type=['pdf', 'csv', 'txt'],
            help="Upload PDF, CSV, or TXT files for analysis"
        )
        
        if uploaded_file:
            # File info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"**Selected:** {uploaded_file.name}\n**Size:** {file_size_mb:.2f} MB\n**Type:** {uploaded_file.type or 'Unknown'}")
            
            # File preview
            if uploaded_file.type == 'text/plain':
                with st.expander("File Preview", expanded=False):
                    content = uploaded_file.read().decode('utf-8')
                    st.text_area("Content", content[:500] + "..." if len(content) > 500 else content, height=150)
                    uploaded_file.seek(0)  # Reset file pointer
        
        # Upload button
        if st.button("Process Document", type="primary", disabled=uploaded_file is None):
            with st.spinner("Processing document..."):
                result = interface.upload_document(uploaded_file)
                
                if result["status"] == "success":
                    st.success(result["message"])
                    
                    # Enhanced success info
                    st.markdown("### Upload Details")
                    st.metric("File", result['filename'])
                    st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                    st.metric("Chunks Created", result.get('chunks_created', 0))
                    
                    # Store document ID in session state
                    st.session_state.current_doc_id = result["document_id"]
                else:
                    st.error(result["message"])
        
        # Sample documents
        with st.expander("Sample Documents", expanded=False):
            st.markdown("""
            **Available sample documents:**
            - `bank_statement.txt` - Account statements and balances
            - `credit_card_transactions.csv` - Transaction data and fees
            - `loan_agreement.txt` - Loan terms and conditions
            """)
            
            # Quick upload buttons for sample docs
            if st.button("Upload Sample Bank Statement"):
                # This would trigger upload of sample document
                st.info("Sample document upload feature coming soon!")
    
    with col2:
        st.markdown("## Intelligent Q&A")
        
        # Sample questions with quick buttons
        with st.expander("Suggested Questions", expanded=False):
            st.markdown("**Quick Questions:**")
            if st.button("What is the account balance?"):
                st.session_state.suggested_question = "What is the account balance?"
            if st.button("Who is the account holder?"):
                st.session_state.suggested_question = "Who is the account holder?"
            if st.button("What are the monthly fees?"):
                st.session_state.suggested_question = "What are the monthly fees?"
            if st.button("Summarize the document"):
                st.session_state.suggested_question = "Summarize the key financial information"
            if st.button("Largest transaction"):
                st.session_state.suggested_question = "What was the largest transaction?"
            if st.button("Credit limit"):
                st.session_state.suggested_question = "What is the credit limit?"
        
        # Question input with character counter and suggestions
        question = st.text_area(
            "Your Question",
            value=st.session_state.get('suggested_question', ''),
            placeholder="Enter your question about the uploaded documents...",
            height=100,
            max_chars=500
        )
        
        if question:
            st.caption(f"Characters: {len(question)}/500")
        
        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            top_k = st.slider("Number of sources", 1, 10, 3)
            use_current_doc = st.checkbox("Use current document only", value=True)
            confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.1)
        
        # Ask button
        if st.button("Ask Question", type="primary", disabled=not question.strip()):
            with st.spinner("Generating response..."):
                # Get document ID
                doc_id = st.session_state.get('current_doc_id', None) if use_current_doc else None
                result = interface.ask_question(question, doc_id)
                
                if result["status"] == "success":
                    # Display answer
                    st.markdown("### AI Response")
                    st.markdown(f"""
                    <div class="response-box">
                        {result["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Performance metrics
                    st.markdown("### Performance Metrics")
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    st.metric("Sources Used", len(result["sources"]))
                    avg_confidence = sum(result["confidence_scores"]) / len(result["confidence_scores"]) if result["confidence_scores"] else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    st.metric("Category", result.get('category', 'Unknown'))
                    
                    # Display sources with enhanced styling
                    st.markdown("### Source Documents")
                    sources = result["sources"]
                    scores = result["confidence_scores"]
                    
                    for i, (source, score) in enumerate(zip(sources, scores), 1):
                        filename = source["metadata"].get("filename", "Unknown")
                        text_preview = source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"]
                        
                        # Color code based on confidence
                        confidence_color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.4 else "#dc3545"
                        
                        st.markdown(f"""
                        <div class="source-item" style="border-left-color: {confidence_color};">
                            <strong>Source {i}</strong> (Relevance: {score:.1%})<br>
                            <em>File: {filename}</em><br>
                            <small>{text_preview}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.error(result["message"])

def render_chat_interface(interface: AdvancedFintechRAGInterface):
    """Render the chat interface"""
    
    st.markdown("## Chat Interface")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown("""
        <div class="chat-container">
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in interface.chat_history[-10:]:  # Show last 10 messages
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message chat-user">
                    <strong>You:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message chat-ai">
                    <strong>AI:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    chat_input = st.text_input("Type your message...", key="chat_input")
    
    if st.button("Send", type="primary"):
        if chat_input.strip():
            with st.spinner("AI is thinking..."):
                result = interface.ask_question(chat_input)
                if result["status"] == "success":
                    st.success("Message sent!")
                    st.rerun()
                else:
                    st.error(result["message"])
    
    if st.button("Clear Chat"):
        interface.chat_history.clear()
        st.success("Chat cleared!")
        st.rerun()

def render_advanced_analytics_dashboard(interface: AdvancedFintechRAGInterface):
    """Render advanced analytics dashboard"""
    
    st.markdown("## Advanced Analytics Dashboard")
    
    if not interface.query_history:
        st.info("No query history available. Start asking questions to see analytics.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", interface.analytics_data['total_queries'])
    with col2:
        success_rate = interface.analytics_data['successful_queries'] / max(interface.analytics_data['total_queries'], 1) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        avg_time = sum(interface.analytics_data['query_times']) / len(interface.analytics_data['query_times']) if interface.analytics_data['query_times'] else 0
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    with col4:
        st.metric("Documents Processed", interface.analytics_data['documents_processed'])
    
    # Charts
    fig_performance, fig_confidence, fig_volume, fig_categories, fig_doc_types = create_advanced_analytics_charts(interface)
    
    # Performance chart
    if fig_performance:
        st.plotly_chart(fig_performance, use_container_width=True)
    
    # Row 1: Confidence and Volume
    col1, col2 = st.columns(2)
    
    with col1:
        if fig_confidence:
            st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col2:
        if fig_volume:
            st.plotly_chart(fig_volume, use_container_width=True)
    
    # Row 2: Categories and Document Types
    col1, col2 = st.columns(2)
    
    with col1:
        if fig_categories:
            st.plotly_chart(fig_categories, use_container_width=True)
    
    with col2:
        if fig_doc_types:
            st.plotly_chart(fig_doc_types, use_container_width=True)
    
    # Query history table with advanced features
    st.markdown("### Recent Queries")
    if interface.query_history:
        df_history = pd.DataFrame(interface.query_history)
        df_display = df_history[['timestamp', 'question', 'processing_time', 'category']].tail(10)
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display.columns = ['Timestamp', 'Question', 'Processing Time (s)', 'Category']
        
        # Add search functionality
        search_term = st.text_input("Search queries...")
        if search_term:
            df_display = df_display[df_display['Question'].str.contains(search_term, case=False, na=False)]
        
        st.dataframe(df_display, use_container_width=True)

def render_advanced_system_health(interface: AdvancedFintechRAGInterface):
    """Render advanced system health monitoring"""
    
    st.markdown("## Advanced System Health Monitoring")
    
    # Health check
    health_result = interface.check_api_health()
    
    if health_result["status"] == "healthy":
        st.success("System Status: Healthy")
        health_data = health_result["data"]
        
        # Component status
        st.markdown("### Component Status")
        for component, status in health_data["components"].items():
            if status:
                st.success(f"✓ {component.replace('_', ' ').title()}")
            else:
                st.error(f"✗ {component.replace('_', ' ').title()}")
        
        # System metrics
        st.markdown("### System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Response Time", f"{health_data.get('response_time', 0):.3f}s")
        with col2:
            st.metric("Uptime", health_data.get('uptime', 'Unknown'))
    else:
        st.error(f"System Status: {health_result['message']}")
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    if interface.analytics_data['query_times']:
        avg_time = sum(interface.analytics_data['query_times']) / len(interface.analytics_data['query_times'])
        min_time = min(interface.analytics_data['query_times'])
        max_time = max(interface.analytics_data['query_times'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Response Time", f"{avg_time:.2f}s")
        with col2:
            st.metric("Fastest Response", f"{min_time:.2f}s")
        with col3:
            st.metric("Slowest Response", f"{max_time:.2f}s")
        with col4:
            st.metric("Total Queries", len(interface.analytics_data['query_times']))
    
    # System resources (simulated)
    st.markdown("### System Resources")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Usage", "45%")
        st.progress(0.45)
    with col2:
        st.metric("Memory Usage", "62%")
        st.progress(0.62)
    with col3:
        st.metric("Disk Usage", "28%")
        st.progress(0.28)

def render_advanced_document_management(interface: AdvancedFintechRAGInterface):
    """Render advanced document management interface"""
    
    st.markdown("## Advanced Document Management")
    
    if not interface.uploaded_docs:
        st.info("No documents uploaded yet.")
        return
    
    # Document list with advanced features
    st.markdown("### Uploaded Documents")
    
    for doc_id, doc_info in interface.uploaded_docs.items():
        with st.expander(f"{doc_info['filename']} (ID: {doc_id})", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Upload Time:** {doc_info['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            with col2:
                st.write(f"**File Size:** {doc_info['file_size']:,} bytes")
            with col3:
                st.write(f"**File Type:** {doc_info['file_type']}")
            with col4:
                st.write(f"**Chunks:** {doc_info.get('chunks_created', 0)}")
            
            # Action buttons
            if st.button(f"View {doc_id[:8]}", key=f"view_{doc_id}"):
                st.info("Document viewer coming soon!")
            if st.button(f"Analyze {doc_id[:8]}", key=f"analyze_{doc_id}"):
                st.info("Document analysis coming soon!")
            if st.button(f"Remove {doc_id[:8]}", key=f"remove_{doc_id}"):
                st.warning("Document removal not yet implemented in API")
    
    # Document statistics
    st.markdown("### Document Statistics")
    if interface.uploaded_docs:
        total_size = sum(doc['file_size'] for doc in interface.uploaded_docs.values())
        avg_size = total_size / len(interface.uploaded_docs)
        total_chunks = sum(doc.get('chunks_created', 0) for doc in interface.uploaded_docs.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(interface.uploaded_docs))
        with col2:
            st.metric("Total Size", f"{total_size:,} bytes")
        with col3:
            st.metric("Average Size", f"{avg_size:,.0f} bytes")
        with col4:
            st.metric("Total Chunks", total_chunks)

def render_settings(interface: AdvancedFintechRAGInterface):
    """Render settings interface"""
    
    st.markdown("## Settings")
    
    # API Configuration
    st.markdown("### API Configuration")
    st.text_input("API Base URL", value=API_BASE_URL, key="api_url")
    st.text_input("API Key", value=API_KEY, type="password", key="api_key")
    
    # Display Settings
    st.markdown("### Display Settings")
    st.checkbox("Show confidence scores", value=True, key="show_confidence")
    st.checkbox("Show processing times", value=True, key="show_times")
    st.checkbox("Auto-refresh analytics", value=True, key="auto_refresh")
    st.slider("Refresh interval (seconds)", 5, 60, 30, key="refresh_interval")
    
    # Export Options
    st.markdown("### Export Options")
    if st.button("Export Query History"):
        if interface.query_history:
            df_export = pd.DataFrame(interface.query_history)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    if st.button("Export Analytics"):
        analytics_data = {
            'total_queries': interface.analytics_data['total_queries'],
            'successful_queries': interface.analytics_data['successful_queries'],
            'avg_confidence': interface.analytics_data['avg_confidence'],
            'documents_processed': interface.analytics_data['documents_processed']
        }
        st.json(analytics_data)

if __name__ == "__main__":
    main() 