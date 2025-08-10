import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="🚀 Advanced RAG Chatbot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rest of imports
import google.generativeai as genai
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import re
import hashlib
from datetime import datetime
import io
import traceback
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for dark theme */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-dark: #0f172a;
        --surface-dark: #1e293b;
        --surface-light: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border-color: #374151;
        --glass-bg: rgba(30, 41, 59, 0.7);
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0;
    }
    
    /* Card styling */
    .custom-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--surface-dark);
        border-right: 1px solid var(--border-color);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        margin-right: 2rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-processing {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* File upload styling */
    .uploadedFile {
        background: var(--glass-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--glass-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(99, 102, 241, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Glassmorphism effect for containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .custom-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_settings(
    temperature: float,
    context_window: int,
    chunk_size: int,
    chunk_overlap: int,
    top_k_results: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate all settings parameters with detailed error messages
    """
    try:
        if not isinstance(temperature, (int, float)):
            return False, "Temperature must be a number"
        if temperature < 0 or temperature > 1:
            return False, "Temperature must be between 0 and 1"

        if not isinstance(context_window, int):
            return False, "Context window must be an integer"
        if context_window < 1:
            return False, "Context window must be at least 1"
        if context_window > 20:
            return False, "Context window cannot exceed 20"

        if not isinstance(chunk_size, int):
            return False, "Chunk size must be an integer"
        if chunk_size < 100:
            return False, "Chunk size must be at least 100 characters"
        if chunk_size > 5000:
            return False, "Chunk size cannot exceed 5000 characters"

        if not isinstance(chunk_overlap, int):
            return False, "Chunk overlap must be an integer"
        if chunk_overlap < 0:
            return False, "Chunk overlap cannot be negative"
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"

        if not isinstance(top_k_results, int):
            return False, "Top K Chunks must be an integer"
        if top_k_results < 1:
            return False, "Top K chunks must be at least 1"
        if top_k_results > 20:
            return False, "Top K chunks cannot exceed 20"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def initialize_session_state():
    """Initialize session state with enhanced error handling and validation"""
    try:
        if 'document_store' not in st.session_state:
            st.session_state.document_store = {}

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {
                'is_processing': False,
                'current_file': None,
                'progress': 0,
                'errors': []
            }

        if 'app_stats' not in st.session_state:
            st.session_state.app_stats = {
                'total_queries': 0,
                'total_documents_processed': 0,
                'session_start_time': datetime.now(),
                'processing_times': []
            }

        default_settings = {
            'model_temperature': 0.7,
            'context_window': 3,
            'chunk_size': 500,
            'chunk_overlap': 100,
            'top_k_results': 5,
            'theme_mode': 'dark'
        }

        for key, value in default_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

        is_valid, error_msg = validate_settings(
            st.session_state.model_temperature,
            st.session_state.context_window,
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
            st.session_state.top_k_results
        )

        if not is_valid:
            logger.warning(f"Invalid settings detected: {error_msg}")
            reset_settings()

    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error("Error initializing application. Please refresh the page.")

def reset_settings():
    """Reset settings to default values with cleanup"""
    settings_keys = [
        'model_temperature',
        'context_window',
        'chunk_size',
        'chunk_overlap',
        'top_k_results'
    ]

    for key in settings_keys:
        if key in st.session_state:
            del st.session_state[key]

class EnhancedPDFProcessor:
    """Enhanced PDF processor with advanced capabilities"""

    def __init__(self, max_chunk_size: int = 1024 * 1024):
        self.max_chunk_size = max_chunk_size

    def process_large_pdf(self, pdf_file: io.BytesIO) -> Dict[str, Any]:
        """Process large PDF files with progress tracking"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)

            if total_pages == 0:
                raise PDFProcessingError("Empty PDF file detected")

            metadata = self._extract_metadata(pdf_reader)
            text_chunks = []

            batch_size = min(20, total_pages)

            for start_idx in range(0, total_pages, batch_size):
                end_idx = min(start_idx + batch_size, total_pages)
                batch_pages = range(start_idx, end_idx)

                with ThreadPoolExecutor() as executor:
                    future_to_page = {
                        executor.submit(self._process_page, pdf_reader.pages[i], i): i
                        for i in batch_pages
                    }

                    for future in as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            page_text = future.result()
                            if page_text.strip():
                                text_chunks.append(f"[Page {page_num + 1}]\n{page_text}")
                        except Exception as e:
                            logger.warning(f"Error processing page {page_num + 1}: {str(e)}")

                progress = (end_idx / total_pages) * 100
                st.session_state.processing_status['progress'] = progress

            return {
                'text_chunks': text_chunks,
                'metadata': metadata
            }

        except Exception as e:
            raise PDFProcessingError(f"Error processing PDF: {str(e)}")

    def _extract_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extract and validate PDF metadata"""
        try:
            metadata = {
                'pages': len(pdf_reader.pages),
                'title': str(pdf_reader.metadata.get('/Title', 'Unknown')) if pdf_reader.metadata else 'Unknown',
                'author': str(pdf_reader.metadata.get('/Author', 'Unknown')) if pdf_reader.metadata else 'Unknown',
                'creation_date': str(pdf_reader.metadata.get('/CreationDate', 'Unknown')) if pdf_reader.metadata else 'Unknown',
                'subject': str(pdf_reader.metadata.get('/Subject', 'Unknown')) if pdf_reader.metadata else 'Unknown',
                'keywords': str(pdf_reader.metadata.get('/Keywords', 'None')) if pdf_reader.metadata else 'None',
                'producer': str(pdf_reader.metadata.get('/Producer', 'Unknown')) if pdf_reader.metadata else 'Unknown',
                'file_size': 'Unknown',
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            for key, value in metadata.items():
                if not isinstance(value, (str, int)) or value is None:
                    metadata[key] = 'Unknown'

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {'error': 'Metadata extraction failed'}

    def _process_page(self, page: PyPDF2.PageObject, page_num: int) -> str:
        """Process a single PDF page with enhanced text extraction"""
        try:
            text = page.extract_text()
            text = self._clean_text(text)

            if not text.strip():
                logger.warning(f"Empty text detected on page {page_num + 1}")
                return ""

            return text

        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {str(e)}")
            return f"[Error processing page {page_num + 1}]"

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better formatting preservation"""
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\w\s.,!?;:()$"\'%-]', '', text)
        text = text.replace('"', '"').replace('"', '"').replace('—', '-')
        return text.strip()

class EnhancedRAGChatbot:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY is not set. Create a .env with GEMINI_API_KEY=<your_key>.")
        genai.configure(api_key=key)
        # Updated model initialization with system instruction for stronger RAG behavior
        self.text_model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=(
                "You are a world-class RAG assistant. Answer strictly based on the provided Document Context and"
                " conversation. If the answer is not in the context, say you don't have enough information from"
                " the uploaded documents. Be concise, structured, and helpful. Use bullet points and short paragraphs."
                " When citing, include [Source: <filename>] where relevant. Do not reveal internal reasoning."
            )
        )
        self.pdf_processor = EnhancedPDFProcessor()

        self.generation_config = {
            "temperature": st.session_state.model_temperature,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Enhanced PDF processing with better error handling"""
        try:
            st.session_state.processing_status['is_processing'] = True
            st.session_state.processing_status['current_file'] = pdf_file.name

            pdf_file.seek(0, 2)
            file_size = pdf_file.tell()
            pdf_file.seek(0)

            if file_size > 100 * 1024 * 1024:
                raise PDFProcessingError("File size exceeds 100MB limit")

            result = self.pdf_processor.process_large_pdf(pdf_file)
            result['metadata']['file_size'] = file_size

            return result

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
            raise

        finally:
            st.session_state.processing_status['is_processing'] = False
            st.session_state.processing_status['current_file'] = None
            st.session_state.processing_status['progress'] = 0

    def chunk_text(self, text_chunks: List[str]) -> List[str]:
        """Enhanced text chunking with better context preservation"""
        chunk_size = st.session_state.chunk_size
        overlap = st.session_state.chunk_overlap

        final_chunks = []

        for text in text_chunks:
            sentences = re.split(r'(?<=[.!?])\s+(?!Page|$Page)', text)

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                if current_length + sentence_length > chunk_size:
                    if current_chunk:
                        final_chunks.append(' '.join(current_chunk))

                    if overlap > 0:
                        overlap_tokens = []
                        overlap_length = 0
                        for s in reversed(current_chunk):
                            if overlap_length + len(s) <= overlap:
                                overlap_tokens.insert(0, s)
                                overlap_length += len(s)
                            else:
                                break
                        current_chunk = overlap_tokens
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0

                current_chunk.append(sentence)
                current_length += sentence_length

            if current_chunk:
                final_chunks.append(' '.join(current_chunk))

        return final_chunks

    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """Enhanced embedding generation with retry logic"""
        embeddings = []
        max_retries = 3

        for chunk in chunks:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if len(chunk) > 8192:
                        chunk = chunk[:8192]

                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=chunk,
                        task_type="retrieval_document"
                    )

                    embedding = np.array(result['embedding'])

                    if embedding.size == 0:
                        raise ValueError("Empty embedding generated")

                    embeddings.append(embedding)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Retry {retry_count} for embedding generation: {str(e)}")

        return embeddings

    def retrieve_relevant_context(self, query: str) -> List[str]:
        """Enhanced context retrieval with better relevance scoring"""
        try:
            processed_query = self.pdf_processor._clean_text(query)

            if st.session_state.conversation_history:
                recent_messages = st.session_state.conversation_history[-st.session_state.context_window:]
                context_text = " ".join([msg['content'] for msg in recent_messages])
                processed_query = f"{context_text} {processed_query}"

            query_embedding = np.array(
                genai.embed_content(
                    model="models/embedding-001",
                    content=processed_query,
                    task_type="retrieval_query"
                )['embedding']
            )

            all_embeddings = []
            all_chunks = []
            chunk_sources = []

            for doc_id, doc_data in st.session_state.document_store.items():
                all_embeddings.extend(doc_data['embeddings'])
                all_chunks.extend(doc_data['chunks'])
                chunk_sources.extend([doc_data['filename']] * len(doc_data['chunks']))

            if not all_embeddings:
                return []

            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                np.array(all_embeddings)
            )[0]

            top_k = min(st.session_state.top_k_results, len(similarities))
            top_indices = similarities.argsort()[-top_k:][::-1]

            min_similarity_threshold = 0.3
            relevant_chunks = []

            for idx in top_indices:
                if similarities[idx] >= min_similarity_threshold:
                    chunk_text = all_chunks[idx]
                    source = chunk_sources[idx]
                    relevant_chunks.append(f"[Source: {source}]\n{chunk_text}")

            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def _compose_prompt(self, query: str, context: List[str]) -> str:
        recent_messages = []
        if st.session_state.conversation_history:
            history_window = st.session_state.conversation_history[-st.session_state.context_window:]
            recent_messages = [
                f"{msg['role']}: {msg['content']}"
                for msg in history_window
            ]

        conversation_context = "\n".join(recent_messages) if recent_messages else "No previous context"
        context_text = "\n\n".join(context) if context else "No relevant context found"

        prompt = f"""
        Previous Conversation:
        {conversation_context}

        Document Context:
        {context_text}

        Current Question: {query}

        Instructions:
        - Be precise, concise, and well-structured.
        - Ground every claim in the Document Context above. If not present, say you lack enough information.
        - Use bullet points where helpful and include brief [Source: <filename>] citations.
        - If the user asks for steps or summaries, provide clear, numbered lists.
        - Maintain consistency with prior answers in this session.
        """
        return prompt

    def generate_response(self, query: str, context: List[str]) -> str:
        """Enhanced response generation with better context utilization"""
        try:
            prompt = self._compose_prompt(query, context)

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    response = self.text_model.generate_content(
                        prompt,
                        generation_config=self.generation_config
                    )

                    if not response or not getattr(response, 'text', None):
                        raise ValueError("Empty response generated")

                    return response.text

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Failed to generate response after {max_retries} attempts: {str(e)}")
                        return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
                    logger.warning(f"Retry {retry_count} for response generation: {str(e)}")

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but an error occurred while generating the response. Please try again."

    def stream_response(self, query: str, context: List[str]):
        """Yield the response incrementally for streaming UI."""
        prompt = self._compose_prompt(query, context)
        try:
            responses = self.text_model.generate_content(
                prompt,
                generation_config=self.generation_config,
                stream=True
            )
            for chunk in responses:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield "\n\nI ran into an error while streaming the response. Please try again."


def create_analytics_dashboard():
    """Create an advanced analytics dashboard"""
    if not st.session_state.document_store:
        st.info("📊 Upload and process documents to see analytics")
        return

    st.markdown("### 📊 Document Analytics Dashboard")
    
    # Prepare data for visualizations
    doc_data = []
    chunk_sizes = []
    page_counts = []
    
    for doc_id, doc_info in st.session_state.document_store.items():
        doc_data.append({
            'filename': doc_info['filename'],
            'chunks': len(doc_info['chunks']),
            'pages': doc_info['metadata']['pages'],
            'file_size_kb': doc_info['metadata']['file_size'] / 1024 if isinstance(doc_info['metadata']['file_size'], (int, float)) else 0,
            'avg_chunk_size': sum(len(chunk) for chunk in doc_info['chunks']) / len(doc_info['chunks']) if doc_info['chunks'] else 0
        })
        chunk_sizes.extend([len(chunk) for chunk in doc_info['chunks']])
        page_counts.append(doc_info['metadata']['pages'])
    
    if doc_data:
        df = pd.DataFrame(doc_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document chunks distribution
            fig_chunks = px.bar(
                df, 
                x='filename', 
                y='chunks',
                title='📄 Chunks per Document',
                color='chunks',
                color_continuous_scale='viridis'
            )
            fig_chunks.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title='Document',
                yaxis_title='Number of Chunks'
            )
            st.plotly_chart(fig_chunks, use_container_width=True)
        
        with col2:
            # File sizes
            fig_size = px.pie(
                df, 
                values='file_size_kb', 
                names='filename',
                title='💾 File Size Distribution (KB)'
            )
            fig_size.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        # Chunk size distribution
        if chunk_sizes:
            fig_dist = px.histogram(
                x=chunk_sizes,
                nbins=20,
                title='📊 Chunk Size Distribution',
                labels={'x': 'Chunk Size (characters)', 'y': 'Frequency'}
            )
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_dist, use_container_width=True)


def create_status_indicator(status: str, text: str) -> str:
    """Create a status indicator with appropriate styling"""
    status_classes = {
        'success': 'status-success',
        'processing': 'status-processing',
        'error': 'status-error'
    }
    
    icons = {
        'success': '✅',
        'processing': '⏳',
        'error': '❌'
    }
    
    class_name = status_classes.get(status, 'status-success')
    icon = icons.get(status, '📄')
    
    return f'<div class="status-indicator {class_name}">{icon} {text}</div>'


def main():
    """Enhanced main function with modern UI"""
    try:
        initialize_session_state()
        load_custom_css()

        api_ready = bool(GEMINI_API_KEY)

        # Main header with glassmorphism effect
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Advanced Document Intelligence</h1>
            <p>Contextual RAG-powered chatbot with enhanced document analysis capabilities</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar Configuration
        with st.sidebar:
            st.markdown("### ⚙️ Configuration Hub")
            
            # API Key Status (no input field)
            if api_ready:
                st.markdown(create_status_indicator('success', 'Using environment API key (🔒 hidden)'), unsafe_allow_html=True)
            else:
                st.markdown(create_status_indicator('error', 'Missing GEMINI_API_KEY in .env'), unsafe_allow_html=True)
                st.info("Create a .env file with GEMINI_API_KEY=<your_key> and restart the app.")

            st.divider()

            # Advanced Settings with enhanced UI
            st.markdown("#### 🔧 Advanced Configuration")
            with st.expander("🎛️ Model Parameters", expanded=False):
                temp_value = st.slider(
                    "🌡️ Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.model_temperature,
                    step=0.1,
                    help="Controls response creativity (0: focused, 1: creative)",
                    key="model_temperature"
                )

                context_value = st.slider(
                    "🧠 Context Window",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.context_window,
                    step=1,
                    help="Number of previous messages to consider",
                    key="context_window"
                )

            with st.expander("📝 Text Processing", expanded=False):
                chunk_value = st.slider(
                    "📏 Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=st.session_state.chunk_size,
                    step=50,
                    help="Size of text segments for processing",
                    key="chunk_size"
                )

                overlap_value = st.slider(
                    "🔗 Chunk Overlap",
                    min_value=0,
                    max_value=min(500, chunk_value - 50),
                    value=min(st.session_state.chunk_overlap, chunk_value - 50),
                    step=10,
                    help="Overlap between consecutive chunks",
                    key="chunk_overlap"
                )

                top_k_value = st.slider(
                    "🎯 Top K Results",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.top_k_results,
                    step=1,
                    help="Number of relevant passages to retrieve",
                    key="top_k_results"
                )

            # Validate settings
            is_valid, error_msg = validate_settings(
                temp_value,
                context_value,
                chunk_value,
                overlap_value,
                top_k_value
            )

            if not is_valid:
                st.error(f"⚠️ {error_msg}")

            st.divider()

            # Document Upload Section with enhanced UI
            st.markdown("#### 📁 Document Upload")
            uploaded_files = st.file_uploader(
                "Select PDF Documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files (max 100MB each)",
                label_visibility="collapsed"
            )

            if uploaded_files:
                st.markdown(f"**📄 {len(uploaded_files)} file(s) selected**")
                for file in uploaded_files:
                    file_size_mb = len(file.getvalue()) / (1024 * 1024)
                    st.markdown(f"• {file.name} ({file_size_mb:.1f} MB)")

            # Process Documents Button with enhanced styling
            col1, col2 = st.columns([3, 1])
            with col1:
                process_button = st.button(
                    "🚀 Process Documents",
                    disabled=not (uploaded_files and api_ready),
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄", help="Refresh", disabled=not uploaded_files):
                    st.rerun()

            # Document Processing with enhanced progress tracking
            if process_button and uploaded_files and api_ready:
                st.markdown("### 🔄 Processing Documents")
                
                try:
                    chatbot = EnhancedRAGChatbot()
                    processed_count = 0
                    errors = []
                    start_time = time.time()

                    # Create progress containers
                    progress_container = st.container()
                    status_container = st.container()

                    with progress_container:
                        overall_progress = st.progress(0)
                        current_file_text = st.empty()

                    with status_container:
                        processing_status = st.empty()

                    for i, pdf_file in enumerate(uploaded_files):
                        try:
                            current_file_text.markdown(f"📄 Processing: **{pdf_file.name}**")
                            processing_status.markdown(
                                create_status_indicator('processing', f'Processing file {i+1} of {len(uploaded_files)}'),
                                unsafe_allow_html=True
                            )

                            # Generate document ID
                            doc_id = hashlib.md5(pdf_file.getvalue()).hexdigest()

                            # Process PDF
                            pdf_data = chatbot.process_pdf(pdf_file)
                            chunks = chatbot.chunk_text(pdf_data['text_chunks'])
                            embeddings = chatbot.generate_embeddings(chunks)

                            # Store processed data
                            st.session_state.document_store[doc_id] = {
                                'filename': pdf_file.name,
                                'chunks': chunks,
                                'embeddings': embeddings,
                                'metadata': pdf_data['metadata']
                            }

                            processed_count += 1
                            
                            # Update progress
                            progress = (i + 1) / len(uploaded_files)
                            overall_progress.progress(progress)

                        except Exception as e:
                            errors.append(f"❌ Error processing {pdf_file.name}: {str(e)}")
                            logger.error(f"Error processing {pdf_file.name}: {str(e)}")

                    # Final status
                    processing_time = time.time() - start_time
                    st.session_state.app_stats['processing_times'].append(processing_time)
                    st.session_state.app_stats['total_documents_processed'] += processed_count

                    current_file_text.empty()
                    processing_status.empty()

                    if processed_count > 0:
                        st.success(f"✅ Successfully processed {processed_count} document(s) in {processing_time:.1f}s!")
                        
                        # Show processing summary
                        total_chunks = sum(len(doc['chunks']) for doc in st.session_state.document_store.values())
                        st.info(f"📊 Generated {total_chunks} text chunks for analysis")

                    if errors:
                        with st.expander("⚠️ Processing Errors", expanded=False):
                            for error in errors:
                                st.error(error)

                except Exception as e:
                    st.error(f"💥 Critical error during processing: {str(e)}")
                    logger.error(f"Critical processing error: {str(e)}\n{traceback.format_exc()}")

            st.divider()

            # Quick Actions Section
            st.markdown("#### ⚡ Quick Actions")
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button("🧹 Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.success("Chat cleared!")
                    time.sleep(1)
                    st.rerun()

            with action_col2:
                if st.button("🗑️ Clear Docs", use_container_width=True):
                    st.session_state.document_store = {}
                    st.success("Documents cleared!")
                    time.sleep(1)
                    st.rerun()

            # Session Statistics
            if st.session_state.document_store or st.session_state.messages:
                st.divider()
                st.markdown("#### 📈 Session Stats")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Documents", len(st.session_state.document_store))
                    st.metric("💬 Messages", len(st.session_state.messages))
                
                with col2:
                    total_chunks = sum(len(doc['chunks']) for doc in st.session_state.document_store.values())
                    st.metric("🧩 Chunks", total_chunks)
                    session_duration = datetime.now() - st.session_state.app_stats['session_start_time']
                    st.metric("⏱️ Session", f"{session_duration.seconds//60}m")

        # Main Content Area
        main_col1, main_col2 = st.columns([2, 1])

        with main_col1:
            # Chat Interface with enhanced styling
            st.markdown("### 💬 Intelligent Chat Interface")
            
            # Chat container with custom styling
            chat_container = st.container()
            
            with chat_container:
                # Display chat messages with enhanced formatting
                for message in st.session_state.messages:
                    with st.chat_message(message["role"], avatar=("🧑\u200d💻" if message["role"] == "user" else "🤖")):
                        if message["role"] == "user":
                            st.markdown(f"""
                            <div class="chat-message user-message">
                                {message["content"]}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="chat-message assistant-message">
                                {message["content"]}
                            </div>
                            """, unsafe_allow_html=True)

            # Chat input with enhanced placeholder
            chat_placeholder = "Ask anything about your documents..." if st.session_state.document_store else "Upload documents first to start chatting"
            
            if prompt := st.chat_input(
                chat_placeholder,
                disabled=not (api_ready and st.session_state.document_store)
            ):
                # Update statistics
                st.session_state.app_stats['total_queries'] += 1
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.conversation_history.append({"role": "user", "content": prompt})

                with st.chat_message("user", avatar="🧑\u200d💻"):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        {prompt}
                    </div>
                    """, unsafe_allow_html=True)

                # Generate and display response (streaming)
                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        chatbot = EnhancedRAGChatbot()
                        
                        # Show processing steps
                        with st.status("🔍 Analyzing your question...", expanded=True) as status:
                            st.write("🧠 Understanding context...")
                            context = chatbot.retrieve_relevant_context(prompt)
                            st.write(f"📚 Found {len(context)} relevant passages")
                            st.write("💭 Generating intelligent response...")
                            status.update(label="🧠 Generating response...", state="running")
                        
                        # Stream the response
                        final_text = st.write_stream(chatbot.stream_response(prompt, context))

                        # Show sources in an elegant expander
                        if context:
                            with st.expander("📚 Source References", expanded=False):
                                sources = {}
                                for i, ctx in enumerate(context):
                                    if ']' in ctx:
                                        source = ctx.split(']')[0].replace('[Source: ', '')
                                        if source not in sources:
                                            sources[source] = []
                                        clean_passage = ctx.split('\n', 1)[1] if '\n' in ctx else ctx
                                        sources[source].append(clean_passage)

                                for source, passages in sources.items():
                                    st.markdown(f"**📄 {source}**")
                                    for j, passage in enumerate(passages, 1):
                                        with st.container():
                                            st.markdown(f"*Excerpt {j}:*")
                                            st.text_area(
                                                label=f"Passage {j}",
                                                value=passage[:500] + "..." if len(passage) > 500 else passage,
                                                height=100,
                                                key=f"source_{source}_{j}",
                                                label_visibility="collapsed",
                                                disabled=True
                                            )

                        # Update conversation history
                        response_text = final_text if isinstance(final_text, str) else ""
                        if not response_text:
                            # Fallback if write_stream didn't return the text
                            response_text = chatbot.generate_response(prompt, context)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
                        
                    except Exception as e:
                        st.error(f"💥 Error generating response: {str(e)}")
                        logger.error(f"Response generation error: {str(e)}\n{traceback.format_exc()}")

        with main_col2:
            # Document Overview Panel
            st.markdown("### 📋 Document Overview")
            
            if st.session_state.document_store:
                # Document cards with enhanced styling
                for doc_id, doc_data in st.session_state.document_store.items():
                    with st.container():
                        st.markdown(f"""
                        <div class="custom-card">
                            <h4>📄 {doc_data['filename']}</h4>
                            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                <span><strong>Pages:</strong> {doc_data['metadata']['pages']}</span>
                                <span><strong>Chunks:</strong> {len(doc_data['chunks'])}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span><strong>Size:</strong> {doc_data['metadata']['file_size'] / 1024:.1f} KB</span>
                                <span><strong>Processed:</strong> {doc_data['metadata']['processing_date']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("📤 Upload documents to see overview")

            # Analytics Dashboard
            st.divider()
            create_analytics_dashboard()

        # Footer with app information
        st.divider()
        
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.markdown("### 🚀 App Features")
            st.markdown("""
            - 🧠 Advanced AI-powered analysis
            - 📚 Multi-document processing
            - 🔍 Intelligent context retrieval
            - 📊 Real-time analytics
            """)
        
        with footer_col2:
            st.markdown("### 📈 Performance Metrics")
            if st.session_state.app_stats['processing_times']:
                avg_processing_time = sum(st.session_state.app_stats['processing_times']) / len(st.session_state.app_stats['processing_times'])
                st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")
            
            st.metric("Total Queries", st.session_state.app_stats['total_queries'])
            st.metric("Documents Processed", st.session_state.app_stats['total_documents_processed'])
        
        with footer_col3:
            st.markdown("### ℹ️ System Info")
            st.markdown(f"""
            - **Session Started:** {st.session_state.app_stats['session_start_time'].strftime('%H:%M:%S')}
            - **Active Documents:** {len(st.session_state.document_store)}
            - **Memory Usage:** Optimized
            - **Status:** 🟢 Online
            """)

    except Exception as e:
        st.error(f"💥 Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        
        # Show error details in expander for debugging
        if st.checkbox("Show error details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()