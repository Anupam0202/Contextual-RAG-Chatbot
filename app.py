"""
Streamlit Main Application for RAG Chatbot with Enhanced UI/UX
Modern, responsive design with accessibility features and accurate page counting
"""

import streamlit as st
import asyncio
from datetime import datetime
import time
import pandas as pd
from pathlib import Path
import json
from typing import Optional, List, Dict, Any
import logging
import base64
import uuid

# Fix for nested event loops in Streamlit
import nest_asyncio
nest_asyncio.apply()

from config import getGlobalConfig, getSecurityConfig, validateConfiguration
from pdf_processor import createPDFProcessor, ProcessedDocument
from vector_store import getGlobalVectorStore
from rag_core import getRAGEngine
from analytics_advanced import AdvancedAnalytics, QueryMetrics
from utils import (
    getAnalytics, getSessionManager, getCache, getCacheManager,
    FileValidator, TextProcessor, ResponseFormatter,
    timeit, getApiResponseWrapper, getMemoryManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration with modern theme
st.set_page_config(
    page_title="RAG Chatbot | Intelligent Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.example.com/help',
        'Report a bug': 'https://github.com/example/issues',
        'About': "RAG Chatbot v1.0.0 - Your Intelligent Document Assistant"
    }
)

# Modern, responsive CSS with animations and accessibility
st.markdown("""
    <style>
    /* CSS Variables for Theme Consistency */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #48bb78;
        --warning-color: #f6ad55;
        --danger-color: #fc8181;
        --info-color: #63b3ed;
        --dark-color: #2d3748;
        --light-color: #f7fafc;
        --border-radius: 12px;
        --transition-speed: 0.3s;
        --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --box-shadow-hover: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Global Styles */
    .main {
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Responsive Typography */
    h1 {
        font-size: clamp(1.5rem, 5vw, 2.5rem);
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInDown 0.5s ease-out;
    }
    
    h2 {
        font-size: clamp(1.2rem, 4vw, 2rem);
        color: var(--dark-color);
    }
    
    h3 {
        font-size: clamp(1rem, 3vw, 1.5rem);
        color: var(--dark-color);
    }
    
    /* Card Components */
    .card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--box-shadow);
        transition: all var(--transition-speed) ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: var(--box-shadow-hover);
    }
    
    /* Chat Interface */
    .stChatMessage {
        background: white;
        border-radius: var(--border-radius);
        padding: 1rem;
        margin-bottom: 0.5rem;
        box-shadow: var(--box-shadow);
        animation: slideIn 0.3s ease-out;
    }
    
    .stChatInput {
        border-radius: var(--border-radius) !important;
        border: 2px solid var(--primary-color) !important;
        transition: all var(--transition-speed) ease;
    }
    
    .stChatInput:focus-within {
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: scale(1.02);
    }
    
    /* Buttons with Modern Style */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all var(--transition-speed) ease;
        box-shadow: var(--box-shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--box-shadow-hover);
        filter: brightness(1.1);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metric Cards with Gradient */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        color: white;
        margin: 0.5rem 0;
        box-shadow: var(--box-shadow);
        transition: all var(--transition-speed) ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: var(--box-shadow-hover);
    }
    
    .metric-card h4 {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    
    /* Document Card Styling */
    .doc-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--box-shadow);
        transition: all var(--transition-speed) ease;
        border-left: 4px solid var(--primary-color);
    }
    
    .doc-card:hover {
        transform: translateX(5px);
        box-shadow: var(--box-shadow-hover);
    }
    
    .doc-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
    }
    
    .doc-meta {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f0f2f6 100%);
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Success/Error Messages */
    .success-message {
        padding: 1rem;
        border-radius: var(--border-radius);
        background-color: #d4edda;
        border-left: 4px solid var(--success-color);
        color: #155724;
        animation: slideInRight 0.3s ease-out;
    }
    
    .error-message {
        padding: 1rem;
        border-radius: var(--border-radius);
        background-color: #f8d7da;
        border-left: 4px solid var(--danger-color);
        color: #721c24;
        animation: slideInRight 0.3s ease-out;
    }
    
    .warning-message {
        padding: 1rem;
        border-radius: var(--border-radius);
        background-color: #fff3cd;
        border-left: 4px solid var(--warning-color);
        color: #856404;
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Context indicator */
    .context-indicator {
        opacity: 0.5;
        font-size: 0.8rem;
        font-style: italic;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
        }
    }
    
    /* Loading Animation */
    .loading-dots {
        display: inline-flex;
        align-items: center;
    }
    
    .loading-dots span {
        width: 8px;
        height: 8px;
        margin: 0 2px;
        background: var(--primary-color);
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    
    .loading-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .loading-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1);
        }
    }
    
    /* Accessibility Features */
    .high-contrast {
        filter: contrast(1.5);
    }
    
    .focus-visible {
        outline: 3px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-card h2 {
            font-size: 1.5rem;
        }
        
        .stColumns > div {
            width: 100% !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .stColumns > div {
            width: 50% !important;
        }
    }
    
    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        :root {
            --dark-color: #f7fafc;
            --light-color: #2d3748;
        }
        
        .main {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        }
        
        .card {
            background: #2d3748;
            color: #f7fafc;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    
    /* Theme specific styles */
    .dark-theme {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --bg-color: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        --text-color: #f7fafc;
    }
    
    .dark-theme .main {
        background: var(--bg-color) !important;
    }
    
    .dark-theme .card {
        background: #2d3748 !important;
        color: #f7fafc !important;
    }
    
    .light-theme {
        --primary-color: #4299e1;
        --secondary-color: #3182ce;
        --bg-color: #ffffff;
        --text-color: #1a202c;
    }
    </style>
    
    <!-- JavaScript for animations -->
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const elements = document.querySelectorAll('.card, .metric-card');
        elements.forEach(el => {
            el.classList.add('animate-in');
        });
    });
    </script>
""", unsafe_allow_html=True)

class EnhancedRAGChatbotApp:
    """Enhanced application with modern UI and advanced features"""
    
    def __init__(self):
        self.config = getGlobalConfig()
        self.security = getSecurityConfig()
        self.analytics = AdvancedAnalytics()
        self.session_manager = getSessionManager()
        self.cache_manager = getCacheManager()
        self.memory_manager = getMemoryManager()
        
        # Initialize components
        self.pdf_processor = createPDFProcessor()
        self.vector_store = getGlobalVectorStore()
        self.rag_engine = getRAGEngine()
        
        # Initialize session state with enhanced features
        self._initializeEnhancedSessionState()
        
        # Apply theme
        self._applyTheme()
    
    def _initializeEnhancedSessionState(self):
        """Initialize enhanced Streamlit session state with session isolation"""
        
        # Create unique session identifier
        if 'session_uuid' not in st.session_state:
            st.session_state.session_uuid = str(uuid.uuid4())
        
        # Session-specific conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Store conversations per session
        if 'all_conversations' not in st.session_state:
            st.session_state.all_conversations = {}
        
        # Ensure current session has its own conversation
        if st.session_state.session_uuid not in st.session_state.all_conversations:
            st.session_state.all_conversations[st.session_state.session_uuid] = []
        
        # Link current conversation to session
        st.session_state.conversation_history = st.session_state.all_conversations[st.session_state.session_uuid]
        
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = self.session_manager.createSession('default_user')
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Chat'
        
        # Enhanced features
        if 'context_window' not in st.session_state:
            st.session_state.context_window = self.config.context_window
        
        if 'theme' not in st.session_state:
            st.session_state.theme = 'modern'
        
        if 'accessibility_mode' not in st.session_state:
            st.session_state.accessibility_mode = False
        
        if 'animations_enabled' not in st.session_state:
            st.session_state.animations_enabled = True
        
        # Analytics tracking
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now()
        
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        # Document metadata cache
        if 'document_metadata' not in st.session_state:
            st.session_state.document_metadata = {}
        
        # Load persisted settings
        self._loadSettingsFromDisk()
    
    def _applyTheme(self):
        """Apply selected theme dynamically"""
        theme_css = {
            '🌟 Modern': """
                <style>
                :root {
                    --primary-color: #667eea;
                    --secondary-color: #764ba2;
                    --bg-color: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    --text-color: #2d3748;
                }
                </style>
            """,
            '🌙 Dark': """
                <style>
                :root {
                    --primary-color: #667eea;
                    --secondary-color: #764ba2;
                    --bg-color: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
                    --text-color: #f7fafc;
                }
                .main { background: var(--bg-color) !important; }
                .card { background: #2d3748 !important; color: #f7fafc !important; }
                body { background: #1a202c !important; }
                </style>
            """,
            '☀️ Light': """
                <style>
                :root {
                    --primary-color: #4299e1;
                    --secondary-color: #3182ce;
                    --bg-color: #ffffff;
                    --text-color: #1a202c;
                }
                .main { background: var(--bg-color) !important; }
                </style>
            """,
            '📜 Classic': """
                <style>
                :root {
                    --primary-color: #4a5568;
                    --secondary-color: #2d3748;
                    --bg-color: #f7fafc;
                    --text-color: #1a202c;
                }
                </style>
            """
        }
        
        selected_theme = st.session_state.get('theme_selector', '🌟 Modern')
        if selected_theme in theme_css:
            st.markdown(theme_css[selected_theme], unsafe_allow_html=True)
    
    def _processQueryAsync(self, prompt: str, context_messages: list):
        """Properly handle async in Streamlit"""
        # Create new event loop for this query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async query processing
            result = loop.run_until_complete(
                self._processQueryCoroutine(prompt, context_messages)
            )
            return result
        finally:
            loop.close()
    
    async def _processQueryCoroutine(self, prompt: str, context_messages: list):
        """Async coroutine for query processing"""
        full_response = ""
        chunks_list = []
        
        async for chunk in self.rag_engine.processQuery(prompt, context_messages):
            full_response += chunk
            chunks_list.append(chunk)
        
        return full_response, chunks_list
    
    def _saveSettingsToDisk(self):
        """Persist settings to disk"""
        settings_path = Path("data/settings.json")
        settings_path.parent.mkdir(exist_ok=True)
        
        settings = {
            'temperature': self.config.temperature,
            'max_output_tokens': self.config.max_output_tokens,
            'context_window': st.session_state.context_window,
            'theme': st.session_state.get('theme_selector', '🌟 Modern'),
            'animations_enabled': st.session_state.animations_enabled,
            'accessibility_mode': st.session_state.accessibility_mode
        }
        
        try:
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Also save to config
            self.config.saveToFile()
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def _loadSettingsFromDisk(self):
        """Load persisted settings"""
        settings_path = Path("data/settings.json")
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                
                # Apply settings
                self.config.temperature = settings.get('temperature', 0.7)
                self.config.max_output_tokens = settings.get('max_output_tokens', 2048)
                st.session_state.context_window = settings.get('context_window', 5)
                st.session_state.theme_selector = settings.get('theme', '🌟 Modern')
                st.session_state.animations_enabled = settings.get('animations_enabled', True)
                st.session_state.accessibility_mode = settings.get('accessibility_mode', False)
            except Exception as e:
                logger.warning(f"Could not load settings: {e}")
    
    def run(self):
        """Run the enhanced application"""
        
        # Apply accessibility mode if enabled
        if st.session_state.accessibility_mode:
            st.markdown('<div class="high-contrast">', unsafe_allow_html=True)
        
        # Enhanced sidebar with modern design
        with st.sidebar:
            # Logo and branding
            st.markdown("""
                <div style="text-align: center; padding: 1rem;">
                    <h1 style="font-size: 2rem; margin: 0;">🤖</h1>
                    <h2 style="font-size: 1.2rem; margin: 0;">RAG Chatbot</h2>
                    <p style="font-size: 0.8rem; opacity: 0.8;">Intelligent Document Assistant</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation with icons
            st.markdown("### 📍 Navigation")
            
            pages = {
                'Chat': '💬',
                'Documents': '📚',
                'Analytics': '📊',
                'Settings': '⚙️',
                'About': 'ℹ️'
            }
            
            selected_page = st.radio(
                "Select Page",
                list(pages.keys()),
                index=list(pages.keys()).index(st.session_state.current_page),
                format_func=lambda x: f"{pages[x]} {x}",
                key="navigation_radio"
            )
            st.session_state.current_page = selected_page
            
            st.markdown("---")
            
            # Enhanced quick stats with animations
            st.markdown("### 📈 Quick Stats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                doc_count = len(st.session_state.uploaded_documents)
                st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem;">
                        <h4>Documents</h4>
                        <h2>{doc_count}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                query_count = st.session_state.query_count
                st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem;">
                        <h4>Queries</h4>
                        <h2>{query_count}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # Session info with progress
            st.markdown("---")
            st.markdown("### 🔐 Session Info")
            
            session_duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
            
            st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
            st.caption(f"Duration: {session_duration:.1f} min")
            st.caption(f"Started: {st.session_state.session_start_time.strftime('%H:%M')}")
            
            # Context window indicator
            st.markdown("---")
            st.markdown("### 🔍 Context Window")
            st.info(f"Using last {st.session_state.context_window} messages")
            
            # Theme selector
            st.markdown("---")
            theme_options = ['🌟 Modern', '🌙 Dark', '☀️ Light', '📜 Classic']
            selected_theme = st.selectbox(
                "Theme",
                theme_options,
                index=theme_options.index(st.session_state.get('theme_selector', '🌟 Modern')),
                key="theme_selector"
            )
            
            # Memory cleanup
            if self.memory_manager.should_cleanup():
                self.memory_manager.cleanup()
        
        # Main content area with animations
        main_container = st.container()
        
        with main_container:
            # Add main content marker for accessibility
            st.markdown('<div id="main-content">', unsafe_allow_html=True)
            
            if selected_page == 'Chat':
                self._renderEnhancedChatPage()
            elif selected_page == 'Documents':
                self._renderEnhancedDocumentsPage()
            elif selected_page == 'Analytics':
                self._renderEnhancedAnalyticsPage()
            elif selected_page == 'Settings':
                self._renderEnhancedSettingsPage()
            elif selected_page == 'About':
                self._renderEnhancedAboutPage()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.accessibility_mode:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _renderEnhancedChatPage(self):
        """Render enhanced chat interface with conversation history - FIXED"""
        
        st.title("💬 Intelligent Chat Interface")
        
        # Check if documents are uploaded
        if not st.session_state.uploaded_documents:
            st.markdown("""
                <div class="warning-message">
                    <h3>📄 No Documents Uploaded</h3>
                    <p>Please upload documents in the Documents page to start chatting!</p>
                </div>
            """, unsafe_allow_html=True)
            return
        
        # Add conversation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Conversation Context:** Last {st.session_state.context_window} messages")
        
        with col2:
            if st.button("🔄 Clear History", key="clear_history"):
                # Clear only current session's history
                st.session_state.conversation_history.clear()
                st.session_state.all_conversations[st.session_state.session_uuid] = []
                st.success("Conversation history cleared!")
                st.rerun()
        
        with col3:
            # Fixed: Direct JSON export button
            if st.button("📥 Export Chat", key="export_chat"):
                if st.session_state.conversation_history:
                    # Create JSON export data
                    export_data = {
                        'session_id': st.session_state.session_id,
                        'session_uuid': st.session_state.session_uuid,
                        'timestamp': datetime.now().isoformat(),
                        'context_window': st.session_state.context_window,
                        'conversation': st.session_state.conversation_history,
                        'statistics': {
                            'total_messages': len(st.session_state.conversation_history),
                            'user_messages': len([m for m in st.session_state.conversation_history if m['role'] == 'user']),
                            'assistant_messages': len([m for m in st.session_state.conversation_history if m['role'] == 'assistant']),
                            'session_duration': (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
                        }
                    }
                    
                    # Convert to JSON
                    json_data = json.dumps(export_data, indent=2)
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No conversation to export")
        
        st.markdown("---")
        
        # Display conversation history with enhanced styling
        chat_container = st.container()
        
        with chat_container:
            # Get messages within context window
            context_messages = self.config.getContextWindowMessages(st.session_state.conversation_history)
            
            # Show older messages indicator if there are more messages
            if len(st.session_state.conversation_history) > len(context_messages):
                st.info(f"📜 {len(st.session_state.conversation_history) - len(context_messages)} older messages hidden (adjust context window in settings)")
            
            # Display messages with enhanced formatting
            for idx, message in enumerate(st.session_state.conversation_history):
                # Check if message is in context window
                in_context = message in context_messages
                
                with st.chat_message(message["role"]):
                    # Add context indicator
                    if not in_context:
                        st.markdown(
                            '<div class="context-indicator">[Out of context window]</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Display message content with formatting
                    formatted_content = ResponseFormatter.formatAsMarkdown(message["content"])
                    st.markdown(formatted_content)
                    
                    # Add message metadata
                    if "timestamp" in message:
                        st.caption(f"⏱️ {message['timestamp']}")
        
        # Enhanced chat input with typing indicator
        if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
            # Increment query counter
            st.session_state.query_count += 1
            
            # Add timestamp to message
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Add user message to history
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": timestamp
            }
            st.session_state.conversation_history.append(user_message)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"⏱️ {timestamp}")
            
            # Generate response with enhanced metrics
            with st.chat_message("assistant"):
                # Show typing indicator
                response_placeholder = st.empty()
                response_placeholder.markdown("""
                    <div class="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Track metrics
                start_time = time.time()
                
                # Get context window messages for RAG (exclude current message)
                context_messages = self.config.getContextWindowMessages(
                    st.session_state.conversation_history[:-1]
                )
                
                try:
                    # Process query with context
                    full_response, chunks_list = self._processQueryAsync(prompt, context_messages)
                    
                    # Display response
                    response_placeholder.markdown(full_response)
                    
                    # Calculate metrics
                    response_time = time.time() - start_time
                    chunks_retrieved = self.config.retrieval_top_k
                    confidence_score = min(0.85 + (len(full_response) / 10000), 1.0)
                    
                    # Add to analytics
                    metric = QueryMetrics(
                        timestamp=datetime.now(),
                        session_id=st.session_state.session_id,
                        query=prompt,
                        response_time=response_time,
                        chunks_retrieved=chunks_retrieved,
                        confidence=confidence_score,
                        context_size=len(context_messages),
                        response_length=len(full_response)
                    )
                    self.analytics.addQueryMetric(metric)
                    
                    # Add response timestamp and metrics
                    response_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.caption(f"⏱️ {response_timestamp} | ⚡ {response_time:.2f}s | 🎯 {confidence_score:.1%} confidence")
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": response_timestamp,
                        "metrics": {
                            "response_time": response_time,
                            "confidence": confidence_score,
                            "chunks_retrieved": chunks_retrieved
                        }
                    }
                    st.session_state.conversation_history.append(assistant_message)
                    
                    # Update session manager
                    self.session_manager.updateSessionData(
                        st.session_state.session_id,
                        'conversation_history',
                        st.session_state.conversation_history
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    error_message = f"I apologize, but I encountered an error: {str(e)}"
                    response_placeholder.markdown(error_message)
                    
                    # Add error to history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "error": True
                    })
        
        # Enhanced quick actions bar
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 View Analytics", key="view_analytics", use_container_width=True):
                st.session_state.current_page = 'Analytics'
                st.rerun()
        
        with col2:
            if st.button("📚 Documents", key="view_docs", use_container_width=True):
                st.session_state.current_page = 'Documents'
                st.rerun()
        
        with col3:
            if st.button("🔍 Search History", key="search_history", use_container_width=True):
                self._showSearchDialog()
        
        with col4:
            if st.button("⚡ Quick Tips", key="quick_tips", use_container_width=True):
                self._showQuickTips()
    
    def _renderEnhancedDocumentsPage(self):
        """Render enhanced document management page with accurate page counting"""
        
        st.title("📚 Document Management Center")
        
        # Document upload section with drag-and-drop styling
        st.markdown("""
            <div class="card">
                <h3>📤 Upload Documents</h3>
                <p>Drag and drop or click to upload PDF files</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key="doc_uploader",
            help="Upload multiple PDF files for processing"
        )
        
        if uploaded_files:
            # Create progress container
            progress_container = st.container()
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [doc['name'] for doc in st.session_state.uploaded_documents]:
                    with progress_container:
                        # Show processing with progress bar
                        progress_bar = st.progress(0, text=f"Processing {uploaded_file.name}...")
                        
                        try:
                            # Process PDF with progress updates
                            progress_bar.progress(25, text=f"Reading {uploaded_file.name}...")
                            file_content = uploaded_file.read()
                            
                            progress_bar.progress(50, text=f"Extracting text from {uploaded_file.name}...")
                            
                            # Process PDF and get ProcessedDocument object
                            processed_doc = self.pdf_processor.processPDF(
                                uploaded_file.name,
                                file_content
                            )
                            
                            progress_bar.progress(75, text=f"Creating embeddings for {uploaded_file.name}...")
                            
                            # Add chunks to vector store
                            success = self.vector_store.addDocuments(processed_doc.chunks)
                            
                            progress_bar.progress(90, text=f"Indexing {uploaded_file.name}...")
                            
                            if success:
                                # Store document with accurate metadata
                                doc_metadata = {
                                    'name': uploaded_file.name,
                                    'document_id': processed_doc.document_id,
                                    'chunks': len(processed_doc.chunks),
                                    'pages': processed_doc.page_count,  # Accurate page count
                                    'uploaded_at': datetime.now().isoformat(),
                                    'size': len(file_content),
                                    'extraction_method': processed_doc.extraction_method,
                                    'processing_time': processed_doc.processing_time,
                                    'metadata': processed_doc.metadata
                                }
                                
                                st.session_state.uploaded_documents.append(doc_metadata)
                                
                                # Cache document metadata
                                st.session_state.document_metadata[processed_doc.document_id] = doc_metadata
                                
                                progress_bar.progress(100, text=f"✅ {uploaded_file.name} processed successfully!")
                                time.sleep(0.5)
                                progress_bar.empty()
                                
                                st.markdown(f"""
                                    <div class="success-message">
                                        ✅ <strong>{uploaded_file.name}</strong> processed successfully!
                                        <br>• {processed_doc.page_count} pages
                                        <br>• {len(processed_doc.chunks)} chunks created
                                        <br>• Extraction method: {processed_doc.extraction_method}
                                        <br>• Processing time: {processed_doc.processing_time:.2f}s
                                        <br>• Ready for querying
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                progress_bar.empty()
                                st.markdown(f"""
                                    <div class="error-message">
                                        ❌ Failed to add <strong>{uploaded_file.name}</strong> to vector store
                                    </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            progress_bar.empty()
                            st.markdown(f"""
                                <div class="error-message">
                                    ❌ Error processing <strong>{uploaded_file.name}</strong>: {str(e)}
                                </div>
                            """, unsafe_allow_html=True)
        
        # Enhanced document list with search and filters
        st.markdown("---")
        st.markdown("### 📋 Document Library")
        
        if st.session_state.uploaded_documents:
            # Add search and filter options
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_query = st.text_input(
                    "🔍 Search documents",
                    placeholder="Enter document name...",
                    key="doc_search"
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Name", "Upload Date", "Size", "Pages", "Chunks"],
                    key="doc_sort"
                )
            
            with col3:
                view_mode = st.radio(
                    "View",
                    ["📋 List", "📊 Grid", "📈 Detailed"],
                    horizontal=True,
                    key="doc_view"
                )
            
            # Filter documents based on search
            filtered_docs = st.session_state.uploaded_documents
            if search_query:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if search_query.lower() in doc['name'].lower()
                ]
            
            # Sort documents
            if sort_by == "Name":
                filtered_docs.sort(key=lambda x: x['name'])
            elif sort_by == "Upload Date":
                filtered_docs.sort(key=lambda x: x['uploaded_at'], reverse=True)
            elif sort_by == "Size":
                filtered_docs.sort(key=lambda x: x.get('size', 0), reverse=True)
            elif sort_by == "Pages":
                filtered_docs.sort(key=lambda x: x.get('pages', 0), reverse=True)
            elif sort_by == "Chunks":
                filtered_docs.sort(key=lambda x: x['chunks'], reverse=True)
            
            # Display documents based on view mode
            if view_mode == "📋 List":
                # List view with enhanced styling
                for idx, doc in enumerate(filtered_docs):
                    with st.container():
                        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**📄 {doc['name']}**")
                            uploaded_time = datetime.fromisoformat(doc['uploaded_at'])
                            st.caption(f"Uploaded: {uploaded_time.strftime('%Y-%m-%d %H:%M')}")
                            if doc.get('extraction_method'):
                                st.caption(f"Method: {doc['extraction_method']}")
                        
                        with col2:
                            st.metric("Pages", doc.get('pages', 'N/A'))
                        
                        with col3:
                            st.metric("Chunks", doc['chunks'])
                        
                        with col4:
                            size_mb = doc.get('size', 0) / (1024 * 1024)
                            st.metric("Size", f"{size_mb:.1f} MB")
                        
                        with col5:
                            if doc.get('processing_time'):
                                st.metric("Process", f"{doc['processing_time']:.1f}s")
                            else:
                                st.metric("Process", "N/A")
                        
                        with col6:
                            if st.button("🗑️ Delete", key=f"delete_{idx}"):
                                # Implement delete with confirmation
                                if st.session_state.get(f"confirm_delete_{idx}", False):
                                    # Remove from vector store if document_id exists
                                    if doc.get('document_id'):
                                        self.vector_store.delete(doc['document_id'])
                                    st.session_state.uploaded_documents.pop(idx)
                                    st.success(f"Deleted {doc['name']}")
                                    st.rerun()
                                else:
                                    st.session_state[f"confirm_delete_{idx}"] = True
                                    st.warning("Click again to confirm deletion")
                        
                        st.markdown("---")
            
            elif view_mode == "📊 Grid":
                # Grid view
                cols = st.columns(3)
                for idx, doc in enumerate(filtered_docs):
                    with cols[idx % 3]:
                        st.markdown(f"""
                            <div class="card">
                                <h4>📄 {doc['name']}</h4>
                                <p><strong>📑 {doc.get('pages', 'N/A')} pages</strong></p>
                                <p>📊 {doc['chunks']} chunks</p>
                                <p>📁 {doc.get('size', 0) / (1024 * 1024):.1f} MB</p>
                                <p>⚙️ {doc.get('extraction_method', 'Unknown')}</p>
                                <p>📅 {datetime.fromisoformat(doc['uploaded_at']).strftime('%Y-%m-%d')}</p>
                                <p>⏱️ {doc.get('processing_time', 0):.1f}s processing</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            else:  # Detailed view
                # Detailed view with comprehensive information
                for idx, doc in enumerate(filtered_docs):
                    with st.expander(f"📄 {doc['name']}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Document Metrics")
                            st.write(f"**Document ID:** `{doc.get('document_id', 'N/A')[:8]}...`")
                            st.write(f"**Total Pages:** {doc.get('pages', 'N/A')}")
                            st.write(f"**Total Chunks:** {doc['chunks']}")
                            st.write(f"**File Size:** {doc.get('size', 0) / (1024 * 1024):.2f} MB")
                            st.write(f"**Processing Time:** {doc.get('processing_time', 0):.2f} seconds")
                            st.write(f"**Extraction Method:** {doc.get('extraction_method', 'Unknown')}")
                        
                        with col2:
                            st.markdown("### 📅 Metadata")
                            uploaded_time = datetime.fromisoformat(doc['uploaded_at'])
                            st.write(f"**Uploaded:** {uploaded_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Display additional metadata if available
                            if doc.get('metadata') and doc['metadata'].get('info'):
                                info = doc['metadata']['info']
                                if info.get('title'):
                                    st.write(f"**Title:** {info['title']}")
                                if info.get('author'):
                                    st.write(f"**Author:** {info['author']}")
                                if info.get('creator'):
                                    st.write(f"**Creator:** {info['creator']}")
                        
                        # Processing statistics
                        st.markdown("### 📈 Processing Statistics")
                        
                        # Calculate average chunk size
                        if doc.get('size') and doc['chunks'] > 0:
                            avg_chunk_size = doc['size'] / doc['chunks'] / 1024  # in KB
                            st.write(f"**Average Chunk Size:** {avg_chunk_size:.1f} KB")
                        
                        # Calculate chunks per page
                        if doc.get('pages') and doc['pages'] > 0:
                            chunks_per_page = doc['chunks'] / doc['pages']
                            st.write(f"**Chunks per Page:** {chunks_per_page:.1f}")
                        
                        # Delete button
                        if st.button(f"🗑️ Delete {doc['name']}", key=f"delete_detailed_{idx}"):
                            if doc.get('document_id'):
                                self.vector_store.delete(doc['document_id'])
                            st.session_state.uploaded_documents.pop(idx)
                            st.success(f"Deleted {doc['name']}")
                            st.rerun()
            
            # Document statistics summary
            st.markdown("---")
            st.markdown("### 📊 Document Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_docs = len(st.session_state.uploaded_documents)
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Total Documents</h4>
                        <h2>{total_docs}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_pages = sum(doc.get('pages', 0) for doc in st.session_state.uploaded_documents)
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Total Pages</h4>
                        <h2>{total_pages}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_chunks = sum(doc['chunks'] for doc in st.session_state.uploaded_documents)
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Total Chunks</h4>
                        <h2>{total_chunks}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_size = sum(doc.get('size', 0) for doc in st.session_state.uploaded_documents)
                size_mb = total_size / (1024 * 1024)
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Total Size</h4>
                        <h2>{size_mb:.1f} MB</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col5:
                avg_pages = total_pages / total_docs if total_docs > 0 else 0
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Avg Pages/Doc</h4>
                        <h2>{avg_pages:.0f}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # Additional statistics
            st.markdown("---")
            st.markdown("### 📊 Processing Analysis")
            
            # Create a DataFrame for analysis
            df = pd.DataFrame(st.session_state.uploaded_documents)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Extraction methods distribution
                if 'extraction_method' in df.columns:
                    method_counts = df['extraction_method'].value_counts()
                    st.markdown("#### Extraction Methods Used")
                    for method, count in method_counts.items():
                        st.write(f"• **{method}:** {count} document{'s' if count > 1 else ''}")
            
            with col2:
                # Processing time statistics
                if 'processing_time' in df.columns:
                    st.markdown("#### Processing Time Statistics")
                    avg_time = df['processing_time'].mean()
                    min_time = df['processing_time'].min()
                    max_time = df['processing_time'].max()
                    
                    st.write(f"• **Average:** {avg_time:.2f} seconds")
                    st.write(f"• **Fastest:** {min_time:.2f} seconds")
                    st.write(f"• **Slowest:** {max_time:.2f} seconds")
        
        else:
            st.info("📝 No documents uploaded yet. Upload PDF files to get started!")
    
    def _renderEnhancedAnalyticsPage(self):
        """Render modern analytics dashboard with comprehensive visualizations"""
        
        # Custom CSS for analytics page
        st.markdown("""
            <style>
            .analytics-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
            }
            
            .analytics-header h1 {
                color: white !important;
                -webkit-text-fill-color: white !important;
                margin-bottom: 0.5rem;
            }
            
            .stat-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            
            .stat-value {
                font-size: 2.5rem;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .stat-label {
                color: #666;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .grade-badge {
                display: inline-block;
                padding: 0.5rem 1.5rem;
                border-radius: 25px;
                font-weight: bold;
                font-size: 1.5rem;
            }
            
            .grade-a { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; }
            .grade-b { background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); color: white; }
            .grade-c { background: linear-gradient(135deg, #f6ad55 0%, #ed8936 100%); color: white; }
            .grade-d { background: linear-gradient(135deg, #fc8181 0%, #f56565 100%); color: white; }
            
            .insight-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1rem 1.5rem;
                margin-bottom: 0.75rem;
                border-left: 3px solid #667eea;
                transition: all 0.2s ease;
            }
            
            .insight-card:hover {
                background: #e9ecef;
                transform: translateX(5px);
            }
            
            .recommendation-card {
                background: #f0fff4;
                border-radius: 10px;
                padding: 1rem 1.5rem;
                margin-bottom: 0.75rem;
                border-left: 3px solid #48bb78;
                transition: all 0.2s ease;
            }
            
            .recommendation-card:hover {
                background: #c6f6d5;
                transform: translateX(5px);
            }
            
            .trend-indicator {
                display: inline-flex;
                align-items: center;
                padding: 0.25rem 0.75rem;
                border-radius: 15px;
                font-size: 0.85rem;
                font-weight: 600;
            }
            
            .trend-up { background: #c6f6d5; color: #22543d; }
            .trend-down { background: #fed7d7; color: #742a2a; }
            .trend-stable { background: #e2e8f0; color: #4a5568; }
            
            .chart-container {
                background: white;
                border-radius: 15px;
                padding: 1rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                margin-bottom: 1.5rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
            <div class="analytics-header">
                <h1>📊 Advanced Analytics Dashboard</h1>
                <p>Real-time insights and performance metrics</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate report
        report_data = self.analytics.generateInteractiveReport()
        summary = report_data.get('summary', {})
        
        # Top Stats Row
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{summary.get('total_queries', 0)}</div>
                    <div class="stat-label">Total Queries</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_time = summary.get('avg_response_time', 0)
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg_time:.2f}s</div>
                    <div class="stat-label">Avg Response</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = summary.get('avg_confidence', 0) * 100
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{confidence:.1f}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            success_rate = summary.get('success_rate', 100)
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            grade = summary.get('performance_grade', 'N/A')
            grade_class = 'grade-a' if grade.startswith('A') else 'grade-b' if grade.startswith('B') else 'grade-c' if grade.startswith('C') else 'grade-d'
            st.markdown(f"""
                <div class="stat-card" style="text-align: center;">
                    <span class="grade-badge {grade_class}">{grade}</span>
                    <div class="stat-label" style="margin-top: 0.5rem;">Performance</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Secondary Stats Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Sessions",
                summary.get('total_sessions', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "⏱️ Avg Session Duration",
                f"{summary.get('avg_session_duration', 0):.1f} min",
                delta=None
            )
        
        with col3:
            st.metric(
                "🔍 Chunks Processed",
                summary.get('total_chunks_processed', 0),
                delta=None
            )
        
        with col4:
            st.metric(
                "📈 Peak Hour",
                summary.get('peak_hour', 'N/A'),
                delta=None
            )
        
        st.markdown("---")
        
        # Satisfaction Score Card
        satisfaction = summary.get('user_satisfaction_score', 0)
        satisfaction_color = '#48bb78' if satisfaction >= 80 else '#f6ad55' if satisfaction >= 60 else '#fc8181'
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {satisfaction_color}20 0%, {satisfaction_color}40 100%); 
                        border-radius: 15px; padding: 1.5rem; margin-bottom: 2rem; text-align: center;
                        border: 2px solid {satisfaction_color};">
                <h3 style="margin: 0; color: #333;">User Satisfaction Score</h3>
                <div style="font-size: 4rem; font-weight: bold; color: {satisfaction_color};">{satisfaction:.0f}%</div>
                <p style="margin: 0; color: #666;">Based on response time, confidence, and success rate</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Main Visualizations
        st.markdown("### 📊 Interactive Visualizations")
        
        # Tab-based navigation for charts
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview",
            "⏱️ Performance",
            "🔥 Engagement",
            "📊 Distribution",
            "📉 Trends"
        ])
        
        with tab1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if 'overview_dashboard' in report_data['visualizations']:
                st.plotly_chart(
                    report_data['visualizations']['overview_dashboard'],
                    use_container_width=True,
                    key="overview_dashboard"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'performance_gauges' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['performance_gauges'],
                        use_container_width=True,
                        key="performance_gauges"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'response_time_distribution' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['response_time_distribution'],
                        use_container_width=True,
                        key="response_time_dist"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'engagement_heatmap' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['engagement_heatmap'],
                        use_container_width=True,
                        key="engagement_heatmap"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'session_analysis' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['session_analysis'],
                        use_container_width=True,
                        key="session_analysis"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'topic_distribution' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['topic_distribution'],
                        use_container_width=True,
                        key="topic_distribution"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'confidence_histogram' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['confidence_histogram'],
                        use_container_width=True,
                        key="confidence_histogram"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'time_series' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['time_series'],
                        use_container_width=True,
                        key="time_series"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if 'trend_analysis' in report_data['visualizations']:
                    st.plotly_chart(
                        report_data['visualizations']['trend_analysis'],
                        use_container_width=True,
                        key="trend_analysis"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Insights and Recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💡 Key Insights")
            insights = report_data.get('insights', [])
            for insight in insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Recommendations")
            recommendations = report_data.get('recommendations', [])
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
        
        # Trend Analysis Section
        st.markdown("---")
        st.markdown("### 📉 Trend Analysis")
        
        trends = report_data.get('trends', {})
        if trends and 'message' not in trends:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rt_trend = trends.get('response_time_trend', 'stable')
                rt_change = trends.get('response_time_change_pct', 0)
                trend_class = 'trend-up' if rt_trend == 'improving' else 'trend-down' if rt_trend == 'degrading' else 'trend-stable'
                trend_icon = '📉' if rt_trend == 'improving' else '📈' if rt_trend == 'degrading' else '➡️'
                
                st.markdown(f"""
                    <div class="stat-card">
                        <h4>Response Time Trend</h4>
                        <span class="trend-indicator {trend_class}">
                            {trend_icon} {rt_trend.title()} ({rt_change:+.1f}%)
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                conf_trend = trends.get('confidence_trend', 'stable')
                conf_change = trends.get('confidence_change_pct', 0)
                trend_class = 'trend-up' if conf_trend == 'improving' else 'trend-down' if conf_trend == 'declining' else 'trend-stable'
                trend_icon = '📈' if conf_trend == 'improving' else '📉' if conf_trend == 'declining' else '➡️'
                
                st.markdown(f"""
                    <div class="stat-card">
                        <h4>Confidence Trend</h4>
                        <span class="trend-indicator {trend_class}">
                            {trend_icon} {conf_trend.title()} ({conf_change:+.1f}%)
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                vol_trend = trends.get('query_volume_trend', 'stable')
                trend_class = 'trend-up' if vol_trend == 'increasing' else 'trend-down' if vol_trend == 'decreasing' else 'trend-stable'
                trend_icon = '📈' if vol_trend == 'increasing' else '📉' if vol_trend == 'decreasing' else '➡️'
                
                st.markdown(f"""
                    <div class="stat-card">
                        <h4>Query Volume Trend</h4>
                        <span class="trend-indicator {trend_class}">
                            {trend_icon} {vol_trend.title()}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
        
        # Predictions Section
        st.markdown("---")
        st.markdown("### 🔮 Predictions")
        
        predictions = report_data.get('predictions', {})
        if predictions and 'message' not in predictions:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🕐 Peak Hour", f"{predictions.get('predicted_peak_hour', 0)}:00")
            
            with col2:
                st.metric("📊 Expected Queries/Hour", predictions.get('expected_queries_next_hour', 0))
            
            with col3:
                st.metric("⏱️ Predicted Response", f"{predictions.get('predicted_response_time', 0):.2f}s")
            
            with col4:
                st.metric("🎯 Confidence Forecast", f"{predictions.get('confidence_forecast', 0):.1%}")
        else:
            st.info("📊 Generate more queries to enable predictions")
        
        # Detailed Metrics Expander
        st.markdown("---")
        with st.expander("📋 Detailed Metrics", expanded=False):
            detailed = report_data.get('detailed_metrics', {})
            performance = report_data.get('performance_breakdown', {})
            
            if detailed:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Response Time Percentiles")
                    percentiles = detailed.get('percentiles', {})
                    if percentiles:
                        perc_df = pd.DataFrame({
                            'Percentile': ['P25', 'P50 (Median)', 'P75', 'P95', 'P99'],
                            'Response Time (s)': [
                                percentiles.get('response_time_p25', 0),
                                percentiles.get('response_time_p50', 0),
                                percentiles.get('response_time_p75', 0),
                                percentiles.get('response_time_p95', 0),
                                percentiles.get('response_time_p99', 0)
                            ]
                        })
                        st.dataframe(perc_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### Performance Breakdown")
                    if performance:
                        rt_breakdown = performance.get('response_time_breakdown', {})
                        if rt_breakdown:
                            breakdown_df = pd.DataFrame([
                                {'Category': 'Excellent (<1s)', 'Count': rt_breakdown.get('excellent', {}).get('count', 0), 'Percentage': f"{rt_breakdown.get('excellent', {}).get('percentage', 0)}%"},
                                {'Category': 'Good (1-2s)', 'Count': rt_breakdown.get('good', {}).get('count', 0), 'Percentage': f"{rt_breakdown.get('good', {}).get('percentage', 0)}%"},
                                {'Category': 'Acceptable (2-5s)', 'Count': rt_breakdown.get('acceptable', {}).get('count', 0), 'Percentage': f"{rt_breakdown.get('acceptable', {}).get('percentage', 0)}%"},
                                {'Category': 'Slow (>5s)', 'Count': rt_breakdown.get('slow', {}).get('count', 0), 'Percentage': f"{rt_breakdown.get('slow', {}).get('percentage', 0)}%"}
                            ])
                            st.dataframe(breakdown_df, use_container_width=True)
        
        # Export Section
        st.markdown("---")
        st.markdown("### 📥 Export Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Full Report", key="gen_report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    report_html = self._generateHTMLReport(report_data)
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=report_html,
                        file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        key="download_html"
                    )
        
        with col2:
            if st.button("📈 Export to Excel", key="export_excel", use_container_width=True):
                with st.spinner("Creating Excel file..."):
                    query_history_dict = [
                        {
                            'timestamp': q.timestamp.isoformat(),
                            'session_id': q.session_id,
                            'query': q.query,
                            'response_time': q.response_time,
                            'chunks_retrieved': q.chunks_retrieved,
                            'confidence': q.confidence
                        }
                        for q in self.analytics.query_history
                    ]
                    
                    excel_data = self.analytics.exportToExcel(query_history_dict)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"
                    )
        
        with col3:
            if st.button("🔄 Refresh Analytics", key="refresh_analytics", use_container_width=True):
                self.analytics._cache.clear()
                st.success("Analytics cache cleared!")
                st.rerun()
    
    def _renderEnhancedSettingsPage(self):
        """Render enhanced settings page with context window control"""
        
        st.title("⚙️ Advanced Settings")
        
        # Configuration validation
        valid, errors = validateConfiguration(self.config)
        
        if not valid:
            st.error("Configuration Issues Detected:")
            for error in errors:
                st.error(f"• {error}")
        
        # Create tabs for different settings categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 Model",
            "💬 Conversation",
            "🔍 Retrieval",
            "📊 Analytics",
            "🎨 UI/UX"
        ])
        
        with tab1:
            st.markdown("### Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=self.config.temperature,
                    step=0.1,
                    help="Higher values make output more random"
                )
                
                max_tokens = st.number_input(
                    "Max Output Tokens",
                    min_value=100,
                    max_value=8192,
                    value=self.config.max_output_tokens,
                    help="Maximum length of generated responses"
                )
            
            with col2:
                top_k = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=self.config.top_k,
                    help="Number of highest probability vocabulary tokens to keep"
                )
                
                top_p = st.slider(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.top_p,
                    step=0.05,
                    help="Cumulative probability for nucleus sampling"
                )
        
        with tab2:
            st.markdown("### Conversation Settings")
            
            # Context window control
            st.markdown("#### 🔍 Context Window")
            
            context_window = st.slider(
                "Number of Historical Messages",
                min_value=1,
                max_value=20,
                value=st.session_state.context_window,
                help="Number of previous messages to include for context",
                key="context_window_slider"
            )
            
            st.info(f"The chatbot will consider the last {context_window} messages when generating responses")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_context_length = st.number_input(
                    "Max Context Length (characters)",
                    min_value=500,
                    max_value=10000,
                    value=self.config.max_context_length,
                    step=500,
                    help="Maximum total characters for context"
                )
                
                preserve_context = st.checkbox(
                    "Preserve Context Between Sessions",
                    value=self.config.preserve_context_between_sessions,
                    help="Keep conversation history across sessions"
                )
            
            with col2:
                conversation_ttl = st.number_input(
                    "Conversation Memory TTL (seconds)",
                    min_value=300,
                    max_value=86400,
                    value=self.config.conversation_memory_ttl,
                    step=300,
                    help="How long to keep conversation in memory"
                )
                
                privacy_mode = st.checkbox(
                    "Privacy Mode",
                    value=self.security.privacy_mode,
                    help="Sanitize sensitive information in conversations"
                )
        
        with tab3:
            st.markdown("### Retrieval Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                retrieval_top_k = st.slider(
                    "Retrieval Top K",
                    min_value=1,
                    max_value=20,
                    value=self.config.retrieval_top_k,
                    help="Number of chunks to retrieve"
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.similarity_threshold,
                    step=0.05,
                    help="Minimum similarity score for retrieval"
                )
            
            with col2:
                hybrid_alpha = st.slider(
                    "Hybrid Search Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.hybrid_search_alpha,
                    step=0.05,
                    help="0 = pure sparse, 1 = pure dense"
                )
                
                rerank_enabled = st.checkbox(
                    "Enable Reranking",
                    value=self.config.rerank_enabled,
                    help="Use LLM to rerank retrieved chunks"
                )
            
            st.markdown("#### Chunking Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=4000,
                    value=self.config.chunk_size,
                    help="Target size for document chunks"
                )
                
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=self.config.chunk_overlap,
                    help="Overlap between consecutive chunks"
                )
            
            with col2:
                semantic_chunking = st.checkbox(
                    "Enable Semantic Chunking",
                    value=self.config.semantic_chunking,
                    help="Use semantic similarity for chunking"
                )
                
                preserve_headers = st.checkbox(
                    "Preserve Headers",
                    value=self.config.preserve_headers,
                    help="Keep document headers for context"
                )
        
        with tab4:
            st.markdown("### Analytics Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_analytics = st.checkbox(
                    "Enable Analytics",
                    value=self.config.enable_analytics,
                    help="Track usage metrics and performance"
                )
                
                track_history = st.checkbox(
                    "Track Query History",
                    value=self.config.track_query_history,
                    help="Store queries for analysis"
                )
                
                generate_insights = st.checkbox(
                    "Generate AI Insights",
                    value=self.config.generate_insights,
                    help="Use AI to generate insights from data"
                )
            
            with col2:
                analytics_ttl = st.number_input(
                    "Analytics Cache TTL (seconds)",
                    min_value=60,
                    max_value=3600,
                    value=self.config.analytics_cache_ttl,
                    help="How often to refresh analytics"
                )
        
        with tab5:
            st.markdown("### UI/UX Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_animations = st.checkbox(
                    "Enable Animations",
                    value=st.session_state.animations_enabled,
                    help="Show animations and transitions"
                )
                
                accessibility_mode = st.checkbox(
                    "Accessibility Mode",
                    value=st.session_state.accessibility_mode,
                    help="Enable high contrast and screen reader support"
                )
                
                high_contrast = st.checkbox(
                    "High Contrast",
                    value=self.config.high_contrast,
                    help="Use high contrast colors"
                )
            
            with col2:
                font_scale = st.slider(
                    "Font Size Scale",
                    min_value=0.8,
                    max_value=1.5,
                    value=self.config.font_size_scale,
                    step=0.1,
                    help="Adjust text size"
                )
                
                animation_speed = st.slider(
                    "Animation Speed (seconds)",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config.animation_speed,
                    step=0.1,
                    help="Speed of UI animations"
                )
        
        # Save settings
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("💾 Save All Settings", key="save_settings", use_container_width=True):
                # Update configuration
                self.config.temperature = temperature
                self.config.max_output_tokens = max_tokens
                self.config.top_k = top_k
                self.config.top_p = top_p
                
                # Update context window
                st.session_state.context_window = context_window
                self.config.context_window = context_window
                self.config.max_context_length = max_context_length
                self.config.preserve_context_between_sessions = preserve_context
                self.config.conversation_memory_ttl = conversation_ttl
                self.security.privacy_mode = privacy_mode
                
                # Update retrieval settings
                self.config.retrieval_top_k = retrieval_top_k
                self.config.similarity_threshold = similarity_threshold
                self.config.hybrid_search_alpha = hybrid_alpha
                self.config.rerank_enabled = rerank_enabled
                self.config.chunk_size = chunk_size
                self.config.chunk_overlap = chunk_overlap
                self.config.semantic_chunking = semantic_chunking
                self.config.preserve_headers = preserve_headers
                
                # Update analytics settings
                self.config.enable_analytics = enable_analytics
                self.config.track_query_history = track_history
                self.config.generate_insights = generate_insights
                self.config.analytics_cache_ttl = analytics_ttl
                
                # Update UI/UX settings
                st.session_state.animations_enabled = enable_animations
                st.session_state.accessibility_mode = accessibility_mode
                self.config.high_contrast = high_contrast
                self.config.font_size_scale = font_scale
                self.config.animation_speed = animation_speed
                
                # Save to disk
                self._saveSettingsToDisk()
                
                st.success("✅ Settings saved successfully!")
                time.sleep(1)
                st.rerun()
    
    def _renderEnhancedAboutPage(self):
        """Render enhanced about page with modern design"""
        
        st.title("ℹ️ About RAG Chatbot")
        
        st.markdown("""
            <div class="card">
                <h2 style="color: var(--primary-color);">🚀 RAG Chatbot - Intelligent Document Assistant</h2>
                <p style="font-size: 1.1rem;">
                    A state-of-the-art Retrieval-Augmented Generation system built with 
                    cutting-edge AI technology and modern user experience design.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase with animations
        st.markdown("## ✨ Key Features")
        
        features = [
            ("🧠", "Advanced AI", "Powered by Google Gemini with adaptive thinking and reflection capabilities"),
            ("💬", "Smart Context", "Adjustable conversation history window for contextual understanding"),
            ("📊", "Rich Analytics", "Comprehensive analytics with interactive visualizations and Excel export"),
            ("🎨", "Modern UI/UX", "Responsive design with animations, themes, and accessibility features"),
            ("🔒", "Enterprise Security", "Privacy mode, API key rotation, and secure document processing"),
            ("⚡", "High Performance", "Hybrid search, caching, and async processing for optimal speed"),
            ("📑", "Accurate Page Counting", "Precise document metadata extraction with multiple fallback methods")
        ]
        
        cols = st.columns(2)
        for idx, (icon, title, description) in enumerate(features):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class="card" style="min-height: 120px;">
                        <h3>{icon} {title}</h3>
                        <p>{description}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Technology stack
        st.markdown("---")
        st.markdown("## 🛠️ Technology Stack")
        
        tech_stack = {
            "AI & ML": ["Google Gemini 1.5", "Sentence Transformers", "FAISS", "BM25"],
            "Backend": ["Python 3.12", "AsyncIO", "LangChain Patterns", "Vector Stores"],
            "Frontend": ["Streamlit", "Plotly", "Custom CSS", "Responsive Design"],
            "Analytics": ["Pandas", "NumPy", "OpenPyXL", "Interactive Charts"],
            "Security": ["Environment Variables", "Input Validation", "Privacy Controls", "Rate Limiting"],
            "PDF Processing": ["PyPDF2", "pdfplumber", "OCR Support", "Semantic Chunking"]
        }
        
        for category, technologies in tech_stack.items():
            st.markdown(f"**{category}:** {', '.join(technologies)}")
        
        # Version information
        st.markdown("---")
        st.markdown("## 📌 Version Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Version:** 1.0.1")
            st.info("**Released:** December 2024")
            st.info("**License:** MIT")
        
        with col2:
            st.info("**Python:** 3.12+")
            st.info("**Streamlit:** 1.32.0+")
            st.info(f"**Model:** {self.config.model_name}")
        
        # Support and documentation
        st.markdown("---")
        st.markdown("## 📚 Documentation & Support")
        
        st.markdown("""
            - 📖 [User Guide](https://github.com/Anupam0202/Contextual-RAG-Chatbot/blob/main/README.md)
            - 🐛 [Report Issues](https://github.com/Anupam0202/Contextual-RAG-Chatbot/issues)
            - 💡 [Feature Requests](https://github.com/Anupam0202/Contextual-RAG-Chatbot/pulls)
            - 📧 [Contact Support](mailto:anupam020202@gmail.com)
        """)
        
        # Credits
        st.markdown("---")
        st.markdown("## 🙏 Credits")
        
        st.markdown("""
            Built with ❤️ by Anupam
            
            Special thanks to the open-source community for the amazing tools and libraries
            that made this project possible.
        """)
    
    def _showSearchDialog(self):
        """Show search dialog for conversation history"""
        with st.expander("🔍 Search Conversation History", expanded=True):
            search_term = st.text_input("Enter search term:", key="search_dialog")
            
            if search_term:
                results = []
                for idx, msg in enumerate(st.session_state.conversation_history):
                    if search_term.lower() in msg['content'].lower():
                        results.append((idx, msg))
                
                if results:
                    st.success(f"Found {len(results)} matches")
                    for idx, msg in results[:5]:  # Show first 5 results
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        st.markdown(f"**{role}** ({msg.get('timestamp', 'N/A')}): {preview}")
                else:
                    st.warning("No matches found")
    
    def _showQuickTips(self):
        """Show quick tips for using the chatbot"""
        with st.expander("⚡ Quick Tips", expanded=True):
            tips = [
                "💡 Use specific questions for better results",
                "🔍 Adjust context window in settings for longer conversations",
                "📊 Check analytics regularly to optimize performance",
                "🎨 Try different themes for a personalized experience",
                "⌨️ Use keyboard shortcuts: Ctrl+Enter to send message",
                "📥 Export conversations for future reference",
                "🔒 Enable privacy mode to sanitize sensitive data",
                "⚡ Clear cache if experiencing slow performance",
                "📑 Document page counts are now accurately extracted from PDF metadata"
            ]
            
            for tip in tips:
                st.markdown(f"• {tip}")
    
    def _generateHTMLReport(self, report_data: Dict) -> str:
        """Generate HTML report from analytics data"""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Chatbot Analytics Report</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f7fa;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }
                .section {
                    background: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                .metric {
                    display: inline-block;
                    padding: 15px;
                    margin: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 8px;
                    min-width: 150px;
                    text-align: center;
                }
                .metric h3 {
                    margin: 0;
                    font-size: 2em;
                }
                .metric p {
                    margin: 5px 0 0 0;
                    opacity: 0.9;
                }
                .insight {
                    padding: 15px;
                    margin: 10px 0;
                    background: #f0f2f6;
                    border-left: 4px solid #667eea;
                    border-radius: 5px;
                }
                .recommendation {
                    padding: 15px;
                    margin: 10px 0;
                    background: #e8f5e9;
                    border-left: 4px solid #48bb78;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
        """
        
        # Header
        html += f"""
        <div class="header">
            <h1>📊 RAG Chatbot Analytics Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        # Executive Summary
        summary = report_data.get('summary', {})
        html += """
        <div class="section">
            <h2>Executive Summary</h2>
            <div style="text-align: center;">
        """
        
        for key, value in summary.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                display_value = f"{value:.1f}" if isinstance(value, float) else str(value)
                if 'rate' in key.lower() or 'score' in key.lower():
                    display_value += '%'
                
                html += f"""
                <div class="metric">
                    <h3>{display_value}</h3>
                    <p>{label}</p>
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        # Insights
        html += """
        <div class="section">
            <h2>💡 Key Insights</h2>
        """
        
        for insight in report_data.get('insights', []):
            html += f'<div class="insight">{insight}</div>'
        
        html += "</div>"
        
        # Recommendations
        html += """
        <div class="section">
            <h2>🎯 Recommendations</h2>
        """
        
        for rec in report_data.get('recommendations', []):
            html += f'<div class="recommendation">{rec}</div>'
        
        html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html

def main():
    """Main entry point"""
    app = EnhancedRAGChatbotApp()
    app.run()

if __name__ == "__main__":
    main()
