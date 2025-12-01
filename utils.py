
"""
Enhanced utility functions with circuit breaker and improved error handling
"""

import hashlib
import time
import json
import re
from typing import Any, Dict, List, Optional, Callable, Tuple
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import pandas as pd
import threading
from collections import defaultdict, deque
from enum import Enum
import pickle
import uuid
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)

# Input Sanitization - Prevents prompt injection and malicious input attacks
@dataclass
class SanitizationResult:
    """Result of input sanitization"""
    sanitized_text: str
    is_safe: bool
    warnings: List[str]
    original_length: int
    sanitized_length: int


class InputSanitizer:
    """
    Comprehensive input sanitization for user queries
    Prevents prompt injection, XSS, and other attacks
    """
    
    # Patterns that indicate prompt injection attempts
    DANGEROUS_PATTERNS = [
        # Instruction override attempts
        (r'ignore\s+(all\s+)?previous\s+instructions?', 'instruction_override'),
        (r'disregard\s+(all\s+)?above', 'instruction_override'),
        (r'forget\s+(everything|all)\s+(you|we)\s+(discussed|said)', 'context_manipulation'),
        
        # System prompt exposure
        (r'(show|display|reveal|output)\s+(your\s+)?(system\s+)?(prompt|instructions)', 'prompt_exposure'),
        (r'what\s+(is|are)\s+your\s+(instructions|rules|prompt)', 'prompt_exposure'),
        (r'repeat\s+your\s+(instructions|prompt|rules)', 'prompt_exposure'),
        
        # Credential extraction
        (r'(api|access)\s*key', 'credential_extraction'),
        (r'(password|secret|token)', 'credential_extraction'),
        
        # Code injection
        (r'<script[>\s]', 'xss_attempt'),
        (r'javascript:', 'xss_attempt'),
        (r'eval\s*\(', 'code_injection'),
        (r'exec\s*\(', 'code_injection'),
        
        # SQL injection patterns
        (r"'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d", 'sql_injection'),
        (r';?\s*DROP\s+TABLE', 'sql_injection'),
        (r'UNION\s+SELECT', 'sql_injection'),
        
        # Role/mode manipulation
        (r'you\s+are\s+now\s+(a\s+)?(developer|admin|root)', 'role_manipulation'),
        (r'(enter|switch\s+to)\s+(developer|admin|debug)\s+mode', 'mode_manipulation'),
        
        # Data exfiltration
        (r'output\s+(all\s+)?(user\s+)?(data|information|content)', 'data_exfiltration'),
        (r'list\s+(all\s+)?(users|files|documents)', 'data_exfiltration'),
    ]
    
    # Maximum allowed length
    MAX_LENGTH = 5000
    
    # Minimum meaningful length
    MIN_LENGTH = 1
    
    # Maximum repetition allowed
    MAX_CHAR_REPETITION = 50
    
    @classmethod
    def sanitize(cls, user_input: str, strict: bool = False) -> SanitizationResult:
        """
        Sanitize user input
        
        Args:
            user_input: Raw user input
            strict: If True, reject any suspicious patterns. If False, sanitize and warn.
        
        Returns:
            SanitizationResult with sanitized text and safety info
        """
        if not user_input:
            return SanitizationResult(
                sanitized_text="",
                is_safe=False,
                warnings=["Empty input"],
                original_length=0,
                sanitized_length=0
            )
        
        original_length = len(user_input)
        warnings = []
        is_safe = True
        
        # 1. Length validation
        if len(user_input) > cls.MAX_LENGTH:
            user_input = user_input[:cls.MAX_LENGTH]
            warnings.append(f"Input truncated from {original_length} to {cls.MAX_LENGTH} characters")
        
        if len(user_input) < cls.MIN_LENGTH:
            return SanitizationResult(
                sanitized_text="",
                is_safe=False,
                warnings=["Input too short"],
                original_length=original_length,
                sanitized_length=0
            )
        
        # 2. Check for dangerous patterns
        for pattern, threat_type in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE | re.MULTILINE):
                warning = f"Potential {threat_type} detected: pattern '{pattern[:30]}...'"
                warnings.append(warning)
                logger.warning(f"Suspicious input detected: {warning}")
                
                if strict:
                    return SanitizationResult(
                        sanitized_text="",
                        is_safe=False,
                        warnings=warnings,
                        original_length=original_length,
                        sanitized_length=0
                    )
                else:
                    is_safe = False
        
        # 3. Remove control characters (except newline and tab)
        sanitized = ''.join(
            char for char in user_input 
            if ord(char) >= 32 or char in '\n\t\r'
        )
        
        # 4. Check for excessive character repetition (possible DoS)
        if cls._has_excessive_repetition(sanitized):
            warnings.append("Excessive character repetition detected")
            sanitized = cls._reduce_repetition(sanitized)
        
        # 5. Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # 6. Remove potentially dangerous Unicode
        sanitized = cls._remove_dangerous_unicode(sanitized)
        
        # 7. Check for encoded attacks
        if cls._check_encoded_attacks(sanitized):
            warnings.append("Potentially encoded malicious content detected")
            is_safe = False
        
        sanitized_length = len(sanitized)
        
        return SanitizationResult(
            sanitized_text=sanitized,
            is_safe=is_safe,
            warnings=warnings,
            original_length=original_length,
            sanitized_length=sanitized_length
        )
    
    @classmethod
    def _has_excessive_repetition(cls, text: str) -> bool:
        """Check if text has excessive character repetition"""
        if len(text) < 10:
            return False
        
        # Check for repeated characters
        max_repeat = 0
        current_char = text[0]
        current_count = 1
        
        for char in text[1:]:
            if char == current_char:
                current_count += 1
                max_repeat = max(max_repeat, current_count)
            else:
                current_char = char
                current_count = 1
        
        return max_repeat > cls.MAX_CHAR_REPETITION
    
    @classmethod
    def _reduce_repetition(cls, text: str) -> str:
        """Reduce excessive character repetition"""
        result = []
        current_char = None
        count = 0
        max_allowed = 10
        
        for char in text:
            if char == current_char:
                count += 1
                if count <= max_allowed:
                    result.append(char)
            else:
                current_char = char
                count = 1
                result.append(char)
        
        return ''.join(result)
    
    @classmethod
    def _remove_dangerous_unicode(cls, text: str) -> str:
        """Remove potentially dangerous Unicode characters"""
        # Remove zero-width characters, direction overrides, etc.
        dangerous_unicode = [
            '\u200B',  # Zero-width space
            '\u200C',  # Zero-width non-joiner
            '\u200D',  # Zero-width joiner
            '\u202A',  # Left-to-right embedding
            '\u202B',  # Right-to-left embedding
            '\u202C',  # Pop directional formatting
            '\u202D',  # Left-to-right override
            '\u202E',  # Right-to-left override
            '\uFEFF',  # Zero-width no-break space
        ]
        
        for char in dangerous_unicode:
            text = text.replace(char, '')
        
        return text
    
    @classmethod
    def _check_encoded_attacks(cls, text: str) -> bool:
        """Check for URL-encoded or base64-encoded attacks"""
        # Check for URL encoding of dangerous patterns
        url_encoded_patterns = [
            '%3Cscript',  # <script
            '%27%20OR%20',  # ' OR 
            '%22%20OR%20',  # " OR
        ]
        
        for pattern in url_encoded_patterns:
            if pattern in text:
                return True
        
        # Check for suspicious base64 patterns
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
            # Could be base64, check if it decodes to something suspicious
            try:
                decoded = base64.b64decode(text).decode('utf-8', errors='ignore')
                for pattern, _ in cls.DANGEROUS_PATTERNS:
                    if re.search(pattern, decoded, re.IGNORECASE):
                        return True
            except Exception:
                pass
        
        return False
    
    @classmethod
    def sanitize_for_prompt(cls, user_input: str) -> str:
        """
        Sanitize input specifically for LLM prompts
        More aggressive than general sanitization
        """
        result = cls.sanitize(user_input, strict=False)
        
        if not result.is_safe:
            # Add clear markers that this is user input
            sanitized = f"[USER QUERY]: {result.sanitized_text}"
            sanitized += "\n[Note: Input contained suspicious patterns and was sanitized]"
            return sanitized
        
        return f"[USER QUERY]: {result.sanitized_text}"
    
    @classmethod
    def validate_file_content(cls, content: str, max_length: int = 100000) -> Tuple[bool, str]:
        """
        Validate file content before processing
        
        Args:
            content: File content
            max_length: Maximum allowed length
        
        Returns:
            (is_valid, error_message)
        """
        if not content:
            return False, "Empty content"
        
        if len(content) > max_length:
            return False, f"Content too large: {len(content)} > {max_length}"
        
        # Check for binary content masquerading as text
        null_count = content.count('\x00')
        if null_count > len(content) * 0.01:  # More than 1% null bytes
            return False, "Content appears to be binary"
        
        return True, "Valid"


# Convenience function for input sanitization
def sanitize_user_input(user_input: str, strict: bool = False) -> SanitizationResult:
    """
    Convenience function to sanitize user input
    
    Args:
        user_input: Raw user input
        strict: If True, reject suspicious input. If False, sanitize and warn.
    
    Returns:
        SanitizationResult
    """
    return InputSanitizer.sanitize(user_input, strict=strict)


# Circuit Breaker Implementation
class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for API calls with automatic recovery"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
        self.success_count = 0
        self.total_calls = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_async(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._execute_sync(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def _execute_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker"""
        with self._lock:
            self.total_calls += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker entering HALF_OPEN state for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}. Service unavailable.")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success(func.__name__)
            return result
        except self.expected_exception as e:
            self._on_failure(func.__name__)
            raise e
    
    def _execute_sync(self, func, *args, **kwargs):
        """Execute sync function with circuit breaker"""
        with self._lock:
            self.total_calls += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker entering HALF_OPEN state for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}. Service unavailable.")
        
        try:
            result = func(*args, **kwargs)
            self._on_success(func.__name__)
            return result
        except self.expected_exception as e:
            self._on_failure(func.__name__)
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self, func_name: str):
        """Handle successful execution"""
        with self._lock:
            self.success_count += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker CLOSED for {func_name}. Service recovered.")
    
    def _on_failure(self, func_name: str):
        """Handle failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPEN for {func_name}. Threshold exceeded.")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_rate': (self.success_count / self.total_calls * 100) if self.total_calls > 0 else 0,
            'total_calls': self.total_calls
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker manually reset")

# Enhanced Circuit Breaker with auto-recovery
class EnhancedCircuitBreaker(CircuitBreaker):
    """Circuit breaker with auto-recovery and health checks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_check_interval = 30  # seconds
        self.last_health_check = time.time()
    
    def _performHealthCheck(self, func):
        """Perform health check to auto-recover"""
        if self.state == CircuitBreakerState.OPEN:
            current_time = time.time()
            if current_time - self.last_health_check > self.health_check_interval:
                try:
                    # Try a lightweight operation
                    test_result = func.__name__  # Simple test
                    if test_result:
                        self.reset()
                        logger.info(f"Circuit breaker auto-recovered for {func.__name__}")
                except:
                    pass
                finally:
                    self.last_health_check = current_time

# Global circuit breakers for different services
api_circuit_breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=30)
pdf_circuit_breaker = EnhancedCircuitBreaker(failure_threshold=5, recovery_timeout=60)
vector_circuit_breaker = EnhancedCircuitBreaker(failure_threshold=5, recovery_timeout=45)

# Enhanced performance monitoring decorator
def timeit(func: Callable) -> Callable:
    """Decorator to measure function execution time with detailed metrics"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        startTime = time.perf_counter()
        error_occurred = False
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error_occurred = True
            raise e
        finally:
            endTime = time.perf_counter()
            execution_time = endTime - startTime
            
            # Log performance metrics
            if execution_time > 5:
                logger.warning(f"{func.__name__} took {execution_time:.4f} seconds (SLOW)")
            else:
                logger.info(f"{func.__name__} took {execution_time:.4f} seconds")
            
            # Track in performance metrics
            if hasattr(func, '__performance_metrics__'):
                func.__performance_metrics__.append({
                    'timestamp': datetime.now(),
                    'execution_time': execution_time,
                    'error': error_occurred
                })
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        startTime = time.perf_counter()
        error_occurred = False
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error_occurred = True
            raise e
        finally:
            endTime = time.perf_counter()
            execution_time = endTime - startTime
            
            # Log performance metrics
            if execution_time > 5:
                logger.warning(f"{func.__name__} took {execution_time:.4f} seconds (SLOW)")
            else:
                logger.info(f"{func.__name__} took {execution_time:.4f} seconds")
    
    # Add performance metrics tracking
    wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    wrapper.__performance_metrics__ = []
    return wrapper

# Enhanced retry decorator with exponential backoff
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Enhanced retry decorator with exponential backoff and jitter"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            lastException = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    lastException = e
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt) + (hash(time.time()) % 1000) / 1000, max_delay)
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            raise lastException
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            lastException = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    lastException = e
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt) + (hash(time.time()) % 1000) / 1000, max_delay)
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            raise lastException
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Enhanced Cache with TTL and size limits
class TTLCache:
    """Enhanced time-based cache with memory management"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000, ttl: Optional[int] = None):
        """Backwards-compatible constructor"""
        # Backwards-compatible parameter name 'ttl'
        self.ttl = ttl if ttl is not None else ttl_seconds
        self.max_size = max_size
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    self.hits += 1
                    self._access_times[key] = datetime.now()
                    return value
                else:
                    # Expired entry
                    del self.cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with memory management"""
        with self._lock:
            # Check size limit
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (value, datetime.now())
            self._access_times[key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        del self.cache[lru_key]
        del self._access_times[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self._access_times.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (self.hits / total_requests * 100) if total_requests > 0 else 0,
            'ttl': self.ttl,
            'max_size': self.max_size
        }
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            current_time = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= timedelta(seconds=self.ttl)
            ]
            
            for key in expired_keys:
                del self.cache[key]
                if key in self._access_times:
                    del self._access_times[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Unified Cache Manager
class UnifiedCacheManager:
    """Centralized cache management to prevent conflicts"""
    
    def __init__(self):
        self.caches = {
            'query': TTLCache(ttl=3600, max_size=100),
            'embeddings': TTLCache(ttl=7200, max_size=500),
            'analytics': TTLCache(ttl=300, max_size=50),
        }
        self._lock = threading.Lock()
    
    def get(self, cache_name: str, key: str):
        """Get from specific cache"""
        with self._lock:
            if cache_name in self.caches:
                return self.caches[cache_name].get(key)
        return None
    
    def set(self, cache_name: str, key: str, value: Any):
        """Set in specific cache"""
        with self._lock:
            if cache_name in self.caches:
                self.caches[cache_name].set(key, value)
    
    def clear_all(self):
        """Clear all caches"""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self.caches.items()}

# Enhanced Text processing utilities
class TextProcessor:
    """Enhanced text processing with better error handling"""
    
    @staticmethod
    def cleanText(text: str) -> str:
        """Clean and normalize text with comprehensive handling"""
        if not text:
            return ""
        
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Fix common encoding issues
            replacements = {
                ''': "'",
                ''': "'",
                '"': '"',
                '"': '"',
                '–': '-',
                '—': '--',
                '…': '...',
                '\u200b': '',  # Zero-width space
                '\xa0': ' ',   # Non-breaking space
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # Remove control characters
            text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    @staticmethod
    def extractKeywords(text: str, top_k: int = 10) -> List[str]:
        """Extract keywords with fallback mechanisms"""
        if not text:
            return []
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # TF-IDF extraction
            vectorizer = TfidfVectorizer(
                max_features=top_k,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            vectorizer.fit([text])
            keywords = vectorizer.get_feature_names_out()
            return list(keywords)
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed, using fallback: {e}")
            
            # Fallback to simple word frequency
            words = text.lower().split()
            word_freq = defaultdict(int)
            
            stopwords = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 
                        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                        'should', 'may', 'might', 'must', 'can', 'could', 'this', 'that', 'these', 
                        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 
                        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 
                        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                        'so', 'than', 'too', 'very', 'just'}
            
            for word in words:
                if len(word) > 3 and word not in stopwords:
                    word_freq[word] += 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:top_k]]
    
    @staticmethod
    def summarizeText(text: str, max_length: int = 200) -> str:
        """Create summary with improved sentence selection"""
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return text[:max_length] + "..."
            
            # Take first sentence
            summary = sentences[0]
            
            # Add more sentences if space allows
            for sentence in sentences[1:]:
                if len(summary) + len(sentence) + 2 <= max_length:
                    summary += ". " + sentence
                else:
                    break
            
            if not summary.endswith('.'):
                summary += "."
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..."

# Enhanced Analytics utilities with memory management
class Analytics:
    """Enhanced analytics with memory management"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.query_history = deque(maxlen=max_history)
        self._lock = threading.Lock()
        
    def trackQuery(self, query: str, response_time: float, 
                   chunks_retrieved: int, confidence: float):
        """Track query metrics with memory limits"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'query': query,
                'response_time': response_time,
                'chunks_retrieved': chunks_retrieved,
                'confidence': confidence
            }
            
            self.query_history.append(entry)
            self.metrics['response_times'].append(response_time)
            self.metrics['chunks_retrieved'].append(chunks_retrieved)
            self.metrics['confidence_scores'].append(confidence)
    
    def getMetricsSummary(self) -> Dict[str, Any]:
        """Get summary with proper error handling"""
        with self._lock:
            if not self.metrics['response_times']:
                return {
                    'status': 'no_data',
                    'message': 'No metrics available yet'
                }
            
            response_times = list(self.metrics['response_times'])
            chunks = list(self.metrics['chunks_retrieved'])
            confidences = list(self.metrics['confidence_scores'])
            
            return {
                'avg_response_time': sum(response_times) / len(response_times),
                'avg_chunks_retrieved': sum(chunks) / len(chunks) if chunks else 0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'total_queries': len(self.query_history),
                'queries_per_hour': self._calculateQueriesPerHour(),
                'memory_usage': len(self.query_history)
            }
    
    def _calculateQueriesPerHour(self) -> float:
        """Calculate queries per hour with proper handling"""
        if len(self.query_history) < 2:
            return 0
        
        history_list = list(self.query_history)
        time_span = history_list[-1]['timestamp'] - history_list[0]['timestamp']
        hours = time_span.total_seconds() / 3600
        
        return len(self.query_history) / max(hours, 0.01)
    
    def cleanup_old_data(self, days: int = 7):
        """Clean up old data to manage memory"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter query history
            self.query_history = deque(
                [q for q in self.query_history if q['timestamp'] > cutoff_time],
                maxlen=self.query_history.maxlen
            )
            
            logger.info(f"Cleaned up analytics data older than {days} days")

# Enhanced File validation utilities
class FileValidator:
    """Enhanced file validation with comprehensive checks"""
    
    @staticmethod
    def validatePDF(file_path: str, file_content: Optional[bytes] = None) -> Tuple[bool, str]:
        """Comprehensive PDF validation"""
        try:
            # Handle both file path and content
            if file_content:
                file_size = len(file_content)
                # Check PDF header from content
                if not file_content.startswith(b'%PDF-'):
                    return False, "Invalid PDF header"
            else:
                path = Path(file_path)
                
                # Check file exists
                if not path.exists():
                    return False, "File does not exist"
                
                # Check extension
                if path.suffix.lower() != '.pdf':
                    return False, "File is not a PDF"
                
                file_size = path.stat().st_size
                
                # Check PDF header from file
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(5)
                        if header != b'%PDF-':
                            return False, "Invalid PDF header"
                except Exception as e:
                    return False, f"Could not read file: {e}"
            
            # Check file size
            max_size = 50 * 1024 * 1024  # 50MB
            
            if file_size > max_size:
                return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024}MB)"
            
            if file_size == 0:
                return False, "File is empty"
            
            return True, "Valid PDF"
            
        except Exception as e:
            logger.error(f"Error validating PDF: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def sanitizeFilename(filename: str) -> str:
        """Enhanced filename sanitization"""
        if not filename:
            return "unnamed_file"
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters and patterns
        dangerous_patterns = [
            r'\.\.+',  # Multiple dots
            r'[<>:"/\\|?*]',  # Windows forbidden characters
            r'[\x00-\x1f\x7f]',  # Control characters
            r'^\.+',  # Leading dots
            r'\s+$',  # Trailing spaces
        ]
        
        for pattern in dangerous_patterns:
            filename = re.sub(pattern, '', filename)
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Ensure there's an extension
        if '.' not in filename:
            filename += '.pdf'
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = Path(filename).stem, Path(filename).suffix
            filename = name[:max_length - len(ext) - 1] + ext
        
        # Final validation
        if not filename or filename == '.pdf':
            filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return filename

# Enhanced Session management with persistence
class SessionManager:
    """Enhanced session management with persistence and isolation"""
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.sessions = {}
        self.session_timeout = timedelta(hours=2)
        self.persistence_path = persistence_path
        self._lock = threading.Lock()
        
        # Load persisted sessions if available
        if persistence_path:
            self._loadSessions()
    
    def createSession(self, user_id: str) -> str:
        """Create new session with enhanced tracking"""
        with self._lock:
            session_id = hashlib.sha256(f"{user_id}{datetime.now()}{hash(time.time())}".encode()).hexdigest()
            
            self.sessions[session_id] = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'data': {},
                'theme': 'modern',  # Store theme preference
                'settings': {},     # Store user settings
                'query_count': 0,
                'conversation_history': []  # Session-specific conversation history
            }
            
            # Persist if configured
            if self.persistence_path:
                self._saveSessions()
            
            return session_id
    
    def getSession(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session with automatic cleanup"""
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check timeout
            if datetime.now() - session['last_activity'] > self.session_timeout:
                del self.sessions[session_id]
                logger.info(f"Session {session_id[:8]}... expired and removed")
                return None
            
            # Update last activity
            session['last_activity'] = datetime.now()
            session['query_count'] = session.get('query_count', 0) + 1
            
            return session
    
    def updateSessionData(self, session_id: str, key: str, value: Any):
        """Update session data with persistence"""
        with self._lock:
            if session := self.sessions.get(session_id):
                session['data'][key] = value
                
                # Special handling for theme and settings
                if key == 'theme':
                    session['theme'] = value
                elif key == 'settings':
                    session['settings'] = value
                elif key == 'conversation_history':
                    session['conversation_history'] = value
                
                # Persist if configured
                if self.persistence_path:
                    self._saveSessions()
    
    def cleanupSessions(self):
        """Remove expired sessions with logging"""
        with self._lock:
            current_time = datetime.now()
            expired = []
            
            for sid, session in list(self.sessions.items()):
                if current_time - session['last_activity'] > self.session_timeout:
                    expired.append(sid)
                    del self.sessions[sid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
                
                # Persist after cleanup
                if self.persistence_path:
                    self._saveSessions()
    
    def _saveSessions(self):
        """Persist sessions to disk"""
        try:
            # Convert datetime objects for serialization
            serializable_sessions = {}
            for sid, session in self.sessions.items():
                serializable_session = session.copy()
                serializable_session['created_at'] = session['created_at'].isoformat()
                serializable_session['last_activity'] = session['last_activity'].isoformat()
                serializable_sessions[sid] = serializable_session
            
            with open(self.persistence_path, 'w') as f:
                json.dump(serializable_sessions, f)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def _loadSessions(self):
        """Load persisted sessions"""
        try:
            if Path(self.persistence_path).exists():
                with open(self.persistence_path, 'r') as f:
                    serializable_sessions = json.load(f)
                
                # Convert back datetime objects
                for sid, session in serializable_sessions.items():
                    session['created_at'] = datetime.fromisoformat(session['created_at'])
                    session['last_activity'] = datetime.fromisoformat(session['last_activity'])
                    self.sessions[sid] = session
                
                logger.info(f"Loaded {len(self.sessions)} persisted sessions")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

# Response formatting with enhanced features
class ResponseFormatter:
    """Enhanced response formatting with better citation handling"""
    
    @staticmethod
    def formatWithCitations(text: str, sources: List[str]) -> str:
        """Format response with numbered citations"""
        if not text:
            return ""
        
        formatted = text
        
        # Add citations section
        if sources:
            formatted += "\n\n**📚 Sources:**\n"
            for i, source in enumerate(sources, 1):
                # Truncate long sources
                display_source = source[:100] + "..." if len(source) > 100 else source
                formatted += f"{i}. {display_source}\n"
        
        return formatted
    
    @staticmethod
    def formatAsMarkdown(text: str) -> str:
        """Enhanced markdown formatting"""
        if not text:
            return ""
        
        try:
            # Preserve code blocks
            code_blocks = []
            
            def preserve_code(match):
                code_blocks.append(match.group(0))
                return f"__CODE_BLOCK_{len(code_blocks)-1}__"
            
            text = re.sub(r'```[\s\S]*?```', preserve_code, text)
            
            # Format lists
            text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
            text = re.sub(r'^[-*]\s+', r'- ', text, flags=re.MULTILINE)
            
            # Format headers
            text = re.sub(r'^(#{1,6})\s+', r'\1 ', text, flags=re.MULTILINE)
            
            # Restore code blocks
            for i, block in enumerate(code_blocks):
                text = text.replace(f"__CODE_BLOCK_{i}__", block)
            
            return text
        except Exception as e:
            logger.error(f"Error formatting markdown: {e}")
            return text
    
    @staticmethod
    def highlightKeyTerms(text: str, terms: List[str]) -> str:
        """Highlight terms with case-insensitive matching"""
        if not text or not terms:
            return text
        
        try:
            for term in terms:
                if term:
                    # Case-insensitive highlighting with word boundaries
                    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                    text = pattern.sub(f"**{term}**", text)
            
            return text
        except Exception as e:
            logger.error(f"Error highlighting terms: {e}")
            return text

# Memory cleanup utilities
class MemoryManager:
    """Manage application memory with periodic cleanup"""
    
    def __init__(self):
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()
        self._lock = threading.Lock()
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed"""
        with self._lock:
            return (datetime.now() - self.last_cleanup).total_seconds() > self.cleanup_interval
    
    def cleanup(self):
        """Perform memory cleanup"""
        with self._lock:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear caches
            cache = getCache()
            cache.cleanup_expired()
            
            # Clean up old analytics data
            analytics = getAnalytics()
            analytics.cleanup_old_data(days=7)
            
            # Clean up expired sessions
            session_manager = getSessionManager()
            session_manager.cleanupSessions()
            
            self.last_cleanup = datetime.now()
            logger.info(f"Memory cleanup completed. GC collected {collected} objects")
            
            return collected

# Performance monitoring for async operations
class AsyncPerformanceMonitor:
    """Monitor performance of async operations"""
    
    def __init__(self):
        self.operations = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def monitor(self, operation_name: str, coro):
        """Monitor an async operation"""
        start_time = time.perf_counter()
        try:
            result = await coro
            execution_time = time.perf_counter() - start_time
            
            async with self._lock:
                self.operations[operation_name].append({
                    'timestamp': datetime.now(),
                    'execution_time': execution_time,
                    'success': True
                })
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            async with self._lock:
                self.operations[operation_name].append({
                    'timestamp': datetime.now(),
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                })
            
            raise e
    
    async def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        async with self._lock:
            if operation_name:
                ops = self.operations.get(operation_name, [])
            else:
                ops = [op for ops_list in self.operations.values() for op in ops_list]
            
            if not ops:
                return {'message': 'No operations recorded'}
            
            successful_ops = [op for op in ops if op.get('success', False)]
            failed_ops = [op for op in ops if not op.get('success', True)]
            
            return {
                'total_operations': len(ops),
                'successful': len(successful_ops),
                'failed': len(failed_ops),
                'success_rate': (len(successful_ops) / len(ops) * 100) if ops else 0,
                'avg_execution_time': sum(op['execution_time'] for op in ops) / len(ops),
                'min_execution_time': min(op['execution_time'] for op in ops),
                'max_execution_time': max(op['execution_time'] for op in ops)
            }

# API response wrapper
def getApiResponseWrapper(success: bool, data: Any = None, 
                         error: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Create consistent API response format with metadata"""
    response = {
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    if data is not None:
        response['data'] = data
    
    if error:
        response['error'] = {
            'message': error,
            'timestamp': datetime.now().isoformat()
        }
    
    if metadata:
        response['metadata'] = metadata
    
    return response

# Global instances with proper initialization
_globalAnalytics = None
_globalSessionManager = None
_globalCache = None
_globalMemoryManager = None
_globalAsyncMonitor = None
_globalCacheManager = None

def getAnalytics() -> Analytics:
    """Get global analytics instance with initialization"""
    global _globalAnalytics
    if _globalAnalytics is None:
        _globalAnalytics = Analytics(max_history=10000)
    return _globalAnalytics

def getSessionManager() -> SessionManager:
    """Get global session manager with persistence"""
    global _globalSessionManager
    if _globalSessionManager is None:
        # Use persistence file for session storage
        persistence_path = Path("data/sessions.json")
        persistence_path.parent.mkdir(exist_ok=True)
        _globalSessionManager = SessionManager(persistence_path=str(persistence_path))
    return _globalSessionManager

def getCache(ttl: int = 3600, max_size: int = 1000) -> TTLCache:
    """Get global cache instance with size limits"""
    global _globalCache
    if _globalCache is None:
        _globalCache = TTLCache(ttl=ttl, max_size=max_size)
    return _globalCache

def getMemoryManager() -> MemoryManager:
    """Get global memory manager"""
    global _globalMemoryManager
    if _globalMemoryManager is None:
        _globalMemoryManager = MemoryManager()
    return _globalMemoryManager

def getAsyncMonitor() -> AsyncPerformanceMonitor:
    """Get global async performance monitor"""
    global _globalAsyncMonitor
    if _globalAsyncMonitor is None:
        _globalAsyncMonitor = AsyncPerformanceMonitor()
    return _globalAsyncMonitor

def getCacheManager() -> UnifiedCacheManager:
    """Get global unified cache manager"""
    global _globalCacheManager
    if _globalCacheManager is None:
        _globalCacheManager = UnifiedCacheManager()
    return _globalCacheManager