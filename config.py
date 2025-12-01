"""
Configuration module for RAG Chatbot
Handles environment variables, security settings, and system configuration
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import secrets
from functools import lru_cache
import re

# Initialize module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration with rotation and encryption"""
    
    def __init__(self):
        self.api_key_rotation_interval = timedelta(days=30)
        self.session_timeout = timedelta(hours=2)
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_extensions = {'.pdf', '.txt'}
        self._key_cache = {}
        self.privacy_mode = True  # Enable privacy features by default
        
    def rotateApiKey(self, service: str) -> str:
        """Rotate API keys based on interval"""
        currentTime = datetime.now()
        if service in self._key_cache:
            lastRotation, key = self._key_cache[service]
            if currentTime - lastRotation < self.api_key_rotation_interval:
                return key
        
        # Generate new key rotation token
        newKey = os.getenv(f"{service.upper()}_API_KEY", "")
        self._key_cache[service] = (currentTime, newKey)
        return newKey
    
    def validateFileUpload(self, file_path: str, file_size: int) -> tuple[bool, str]:
        """Validate uploaded files for security"""
        extension = Path(file_path).suffix.lower()
        
        if extension not in self.allowed_extensions:
            return False, f"File type {extension} not allowed"
        
        if file_size > self.max_file_size:
            return False, f"File size exceeds {self.max_file_size / 1024 / 1024:.0f}MB limit"
        
        return True, "Valid"
    
    def sanitizeConversationHistory(self, history: list) -> list:
        """Sanitize conversation history for privacy"""
        if not self.privacy_mode:
            return history
        
        # Remove sensitive information patterns
        sensitive_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b\d{9}\b',  # SSN-like numbers
        ]
        
        sanitized_history = []
        for entry in history:
            sanitized_entry = entry.copy()
            for pattern in sensitive_patterns:
                sanitized_entry['content'] = re.sub(pattern, '[REDACTED]', sanitized_entry.get('content', ''))
            sanitized_history.append(sanitized_entry)
        
        return sanitized_history

@dataclass
class RAGConfig:
    """Main configuration for RAG system"""
    
    # Model Configuration
    model_name: str = "gemini-1.5-flash-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_k: int = 10
    top_p: float = 0.95
    
    # Conversation Configuration
    context_window: int = 5  # Number of historical messages to consider
    max_context_length: int = 4000  # Maximum characters for context
    preserve_context_between_sessions: bool = False
    conversation_memory_ttl: int = 3600  # Time to live for conversation memory (seconds)
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_chunking: bool = True
    preserve_headers: bool = True
    
    # Retrieval Configuration
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    hybrid_search_alpha: float = 0.5
    rerank_enabled: bool = True
    
    # Vector Store Configuration
    vector_store_type: str = "in_memory"
    vector_dimension: int = 384
    index_name: str = "rag_documents"
    batch_size: int = 100
    
    # Analytics Configuration
    enable_analytics: bool = True
    analytics_cache_ttl: int = 300
    track_query_history: bool = True
    generate_insights: bool = True  # Enable AI-powered insights
    
    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600
    async_processing: bool = True
    max_workers: int = 4
    
    # UI/UX Configuration
    theme: str = "modern"  # Options: modern, classic, dark, light
    enable_animations: bool = True
    animation_speed: float = 0.3  # seconds
    accessibility_mode: bool = False
    high_contrast: bool = False
    font_size_scale: float = 1.0
    
    # Safety Configuration
    safety_settings: Dict[str, str] = field(default_factory=lambda: {
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE"
    })
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        config_dict = {}
        
        # Load from .env file if exists
        env_path = Path('.env')
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv()
        
        # Override with environment variables
        for field_name in cls.__dataclass_fields__:
            env_var = f"RAG_{field_name.upper()}"
            if env_value := os.getenv(env_var):
                field_type = cls.__dataclass_fields__[field_name].type
                
                # Type conversion
                if field_type == bool:
                    config_dict[field_name] = env_value.lower() in ('true', '1', 'yes')
                elif field_type == int:
                    config_dict[field_name] = int(env_value)
                elif field_type == float:
                    config_dict[field_name] = float(env_value)
                elif field_type == Dict[str, str]:
                    config_dict[field_name] = json.loads(env_value)
                else:
                    config_dict[field_name] = env_value
        
        return cls(**config_dict)
    
    def getOptimalChunkSize(self, document_length: int) -> int:
        """Dynamic chunk size based on document length"""
        if document_length < 5000:
            return min(500, document_length)
        elif document_length < 20000:
            return 1000
        else:
            return 1500

    def getContextWindowMessages(self, conversation_history: list) -> list:
        """Get messages within the context window - FIXED for proper context handling"""
        if not conversation_history:
            return []
        
        # Get last N messages (counting individual messages, not pairs)
        context_messages = conversation_history[-self.context_window:]
        
        # Check total length and trim if needed
        total_length = sum(len(msg.get('content', '')) for msg in context_messages)
        
        while total_length > self.max_context_length and len(context_messages) > 1:
            # Remove oldest message if context is too long
            context_messages = context_messages[1:]
            total_length = sum(len(msg.get('content', '')) for msg in context_messages)
        
        logger.info(f"Context window: {len(context_messages)} messages, {total_length} characters")
        
        return context_messages
    
    def toDict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def saveToFile(self, path: str = "data/config.json"):
        """Save configuration to file"""
        config_path = Path(path)
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.toDict(), f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {path}")
    
    def loadFromFile(self, path: str = "data/config.json"):
        """Load configuration from file"""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Configuration loaded from {path}")

@lru_cache(maxsize=1)
def getGlobalConfig() -> RAGConfig:
    """Get singleton configuration instance"""
    config = RAGConfig.from_env()
    
    # Try to load saved configuration
    config_path = Path("data/config.json")
    if config_path.exists():
        config.loadFromFile()
    
    return config

@lru_cache(maxsize=1)
def getSecurityConfig() -> SecurityConfig:
    """Get singleton security configuration"""
    return SecurityConfig()

# Configuration validator
def validateConfiguration(config: RAGConfig) -> tuple[bool, list[str]]:
    """Validate configuration settings"""
    errors = []
    
    if config.chunk_size <= config.chunk_overlap:
        errors.append("Chunk size must be greater than overlap")
    
    if config.temperature < 0 or config.temperature > 2:
        errors.append("Temperature must be between 0 and 2")
    
    if config.retrieval_top_k > 100:
        errors.append("Retrieval top_k should not exceed 100")
    
    if config.context_window < 1 or config.context_window > 20:
        errors.append("Context window must be between 1 and 20 messages")
    
    if not os.getenv("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY not found in environment")
    
    return len(errors) == 0, errors
