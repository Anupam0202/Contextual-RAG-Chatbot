"""
Enhanced PDF Processing Module with improved error handling and OCR recovery
"""

import io
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import uuid
from datetime import datetime

import PyPDF2
from pdfplumber import PDF as PlumberPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import getGlobalConfig, getSecurityConfig
from utils import pdf_circuit_breaker, retry_with_backoff, FileValidator

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

@dataclass
class DocumentChunk:
    """Enhanced document chunk with validation"""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    page_number: int = 0
    chunk_index: int = 0
    semantic_score: float = 0.0
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())
        
        # Validate content
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        
        # Add computed metadata
        self.metadata['word_count'] = len(self.content.split())
        self.metadata['char_count'] = len(self.content)
        self.metadata['created_at'] = datetime.now().isoformat()
    
    def is_valid(self) -> bool:
        """Check if chunk is valid"""
        return bool(self.content and self.content.strip() and len(self.content) > 10)

@dataclass
class ProcessedDocument:
    """Container for processed document with metadata"""
    document_id: str
    file_path: str
    chunks: List[DocumentChunk]
    page_count: int
    metadata: Dict[str, Any]
    extraction_method: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'document_id': self.document_id,
            'file_path': self.file_path,
            'chunk_count': len(self.chunks),
            'page_count': self.page_count,
            'metadata': self.metadata,
            'extraction_method': self.extraction_method,
            'processing_time': self.processing_time
        }

class SemanticChunker:
    """Enhanced semantic chunking with error recovery"""
    
    def __init__(self, model_name: str = None):
        """Initialize semantic chunker with error handling"""
        self.config = getGlobalConfig()
        
        # Use configured model or fallback
        if model_name is None:
            model_name = self.config.embedding_model
        
        # Fix Google model references
        if model_name.startswith("models/"):
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Initialized semantic chunker with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}, using fallback: {e}")
            self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
    def extractSemanticSegments(self, text: str) -> List[str]:
        """Extract semantic segments with error handling"""
        if not text or len(text) < 50:
            return [text] if text else []
        
        try:
            # Split into sentences
            sentences = self._splitIntoSentences(text)
            
            if len(sentences) <= 1:
                return [text]
            
            # Get embeddings with batching for efficiency
            batch_size = 100
            embeddings = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                batch_embeddings = self.sentence_model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for idx in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    embeddings[idx + 1].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            # Find breakpoints where similarity is low
            breakpoints = self._findBreakpoints(similarities)
            
            # Create chunks based on breakpoints
            chunks = []
            start_idx = 0
            
            for breakpoint in breakpoints:
                chunk = ' '.join(sentences[start_idx:breakpoint + 1])
                if len(chunk) > 50:  # Minimum chunk size
                    chunks.append(chunk)
                start_idx = breakpoint + 1
            
            # Add remaining sentences
            if start_idx < len(sentences):
                chunk = ' '.join(sentences[start_idx:])
                if chunk:
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to simple chunking: {e}")
            return self._fallbackChunking(text)
    
    def _splitIntoSentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling"""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+|(?<=[.!?])"?\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleanedSentences = []
        for sent in sentences:
            cleaned = sent.strip()
            if cleaned and len(cleaned) > 10:
                cleanedSentences.append(cleaned)
        
        return cleanedSentences if cleanedSentences else [text]
    
    def _findBreakpoints(self, similarities: List[float], threshold: float = 0.3) -> List[int]:
        """Find semantic breakpoints based on similarity scores"""
        if not similarities:
            return []
        
        breakpoints = []
        
        # Dynamic threshold based on similarity distribution
        meanSim = np.mean(similarities)
        stdSim = np.std(similarities)
        dynamicThreshold = max(threshold, meanSim - stdSim)
        
        for idx, sim in enumerate(similarities):
            if sim < dynamicThreshold:
                breakpoints.append(idx)
        
        return breakpoints
    
    def _fallbackChunking(self, text: str) -> List[str]:
        """Fallback to simple chunking when semantic fails"""
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
        
        return chunks

class PDFProcessor:
    """Enhanced PDF processor with comprehensive error handling and accurate page counting"""
    
    def __init__(self):
        self.config = getGlobalConfig()
        self.security = getSecurityConfig()
        self.semantic_chunker = SemanticChunker() if self.config.semantic_chunking else None
        self.processed_documents = {}
        self.processing_errors = []
        self.ocr_available = self._checkOCRAvailability()
        
    def _checkOCRAvailability(self) -> bool:
        """Check if OCR is properly configured"""
        try:
            import pytesseract
            # Test tesseract availability
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"OCR not available: {e}")
            return False
    
    @pdf_circuit_breaker
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def processPDF(self, file_path: str, file_content: Optional[bytes] = None) -> ProcessedDocument:
        """Process PDF with enhanced error handling and accurate page counting"""
        start_time = datetime.now()
        
        try:
            # Enhanced validation
            if file_content:
                valid, message = FileValidator.validatePDF(file_path, file_content)
            else:
                valid, message = FileValidator.validatePDF(file_path)
            
            if not valid:
                raise ProcessingError(f"Validation failed: {message}")
            
            # Generate document ID
            content_for_id = file_content if file_content else open(file_path, 'rb').read()
            document_id = self._generateDocumentId(content_for_id)
            
            # Check cache
            if document_id in self.processed_documents:
                logger.info(f"Document {document_id} already processed, returning cached result")
                return self.processed_documents[document_id]
            
            # Extract content with multiple methods
            extractedContent = self._extractContentWithFallback(file_path, file_content)
            
            if not extractedContent['text']:
                raise ProcessingError("No text could be extracted from PDF")
            
            # Get accurate page count
            page_count = extractedContent['metadata'].get('page_count', 0)
            if page_count == 0:
                # Fallback: count the number of text pages
                page_count = len(extractedContent.get('text', []))
            
            # Create chunks
            chunks = self._createChunksWithValidation(extractedContent, document_id, file_path)
            
            if not chunks:
                raise ProcessingError("No valid chunks could be created")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create ProcessedDocument object
            processed_doc = ProcessedDocument(
                document_id=document_id,
                file_path=file_path,
                chunks=chunks,
                page_count=page_count,
                metadata=extractedContent['metadata'],
                extraction_method=extractedContent.get('extraction_method', 'unknown'),
                processing_time=processing_time
            )
            
            # Cache results
            self.processed_documents[document_id] = processed_doc
            
            logger.info(f"Successfully processed PDF: {file_path}, {page_count} pages, {len(chunks)} chunks")
            return processed_doc
            
        except ProcessingError:
            raise
        except Exception as e:
            error_msg = f"Error processing PDF {file_path}: {str(e)}"
            logger.error(error_msg)
            self.processing_errors.append({
                'file': file_path,
                'error': str(e),
                'timestamp': datetime.now()
            })
            raise ProcessingError(error_msg)
    
    def _generateDocumentId(self, content: bytes) -> str:
        """Generate unique document ID based on content hash"""
        return hashlib.sha256(content).hexdigest()[:16]
    
    def _extractContentWithFallback(self, file_path: str, file_content: Optional[bytes]) -> Dict[str, Any]:
        """Extract content with multiple fallback methods and accurate page counting"""
        extractedData = {
            'text': [],
            'metadata': {},
            'pages': [],
            'extraction_method': 'unknown'
        }
        
        pdf_file = io.BytesIO(file_content) if file_content else None
        
        # Method 1: PyPDF2 (Primary method for accurate page count)
        try:
            text, metadata, pages = self._extractWithPyPDF2(pdf_file or file_path)
            if text or metadata.get('page_count', 0) > 0:
                extractedData['text'] = text
                extractedData['metadata'] = metadata
                extractedData['pages'] = pages
                extractedData['extraction_method'] = 'PyPDF2'
                logger.info(f"Successfully extracted text with PyPDF2, {metadata.get('page_count', 0)} pages")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfplumber for tables and structured content
        if pdf_file:
            pdf_file.seek(0)
        
        try:
            additional_text, plumber_page_count = self._extractWithPdfplumber(pdf_file or file_path)
            
            # Update page count if not already set
            if extractedData['metadata'].get('page_count', 0) == 0:
                extractedData['metadata']['page_count'] = plumber_page_count
            
            if additional_text:
                for i, text in enumerate(additional_text):
                    if i < len(extractedData['text']):
                        extractedData['text'][i] += f"\n\n{text}"
                    else:
                        extractedData['text'].append(text)
                
                if extractedData['extraction_method'] == 'unknown':
                    extractedData['extraction_method'] = 'pdfplumber'
                
                logger.info(f"Enhanced extraction with pdfplumber, confirmed {plumber_page_count} pages")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 3: OCR for scanned documents (if available)
        if self.ocr_available and self._shouldAttemptOCR(extractedData['text']):
            if pdf_file:
                pdf_file.seek(0)
            
            try:
                ocr_text, ocr_page_count = self._performOCRWithValidation(pdf_file or file_path)
                if ocr_text:
                    extractedData['text'] = ocr_text
                    extractedData['extraction_method'] = 'OCR'
                    
                    # Update page count from OCR if needed
                    if extractedData['metadata'].get('page_count', 0) == 0:
                        extractedData['metadata']['page_count'] = ocr_page_count
                    
                    logger.info(f"Successfully extracted text with OCR, {ocr_page_count} pages")
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                if not extractedData['text']:
                    logger.warning("All extraction methods failed. OCR not available or failed.")
        
        # Final validation of page count
        if extractedData['metadata'].get('page_count', 0) == 0 and extractedData['text']:
            extractedData['metadata']['page_count'] = len(extractedData['text'])
        
        return extractedData
    
    def _extractWithPyPDF2(self, pdf_source) -> Tuple[List[str], Dict, List[Dict]]:
        """Extract text using PyPDF2 with accurate page counting"""
        text_pages = []
        metadata = {}
        pages = []
        
        if isinstance(pdf_source, io.BytesIO):
            reader = PyPDF2.PdfReader(pdf_source)
        else:
            reader = PyPDF2.PdfReader(pdf_source)
        
        # Get accurate page count
        page_count = len(reader.pages)
        metadata['page_count'] = page_count
        
        # Extract document info
        if hasattr(reader, 'metadata') and reader.metadata:
            metadata['info'] = {
                'title': getattr(reader.metadata, 'title', None),
                'author': getattr(reader.metadata, 'author', None),
                'subject': getattr(reader.metadata, 'subject', None),
                'creator': getattr(reader.metadata, 'creator', None),
                'producer': getattr(reader.metadata, 'producer', None),
                'creation_date': str(getattr(reader.metadata, 'creation_date', None)),
                'modification_date': str(getattr(reader.metadata, 'modification_date', None)),
            }
        
        # Extract text from each page
        for pageNum, page in enumerate(reader.pages):
            try:
                pageText = page.extract_text()
                
                if self.config.preserve_headers:
                    headers = self._extractHeaders(pageText)
                    pageText = self._enrichTextWithHeaders(pageText, headers)
                else:
                    headers = []
                
                text_pages.append(pageText)
                pages.append({
                    'page_number': pageNum + 1,
                    'text': pageText,
                    'headers': headers,
                    'has_content': bool(pageText and pageText.strip())
                })
            except Exception as e:
                logger.warning(f"Failed to extract page {pageNum + 1}: {e}")
                text_pages.append("")
                pages.append({
                    'page_number': pageNum + 1,
                    'text': "",
                    'headers': [],
                    'error': str(e),
                    'has_content': False
                })
        
        return text_pages, metadata, pages
    
    def _extractWithPdfplumber(self, pdf_source) -> Tuple[List[str], int]:
        """Extract tables and structured content with pdfplumber, return text and page count"""
        extracted_text = []
        page_count = 0
        
        try:
            with PlumberPDF(pdf_source) as pdf:
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    page_text = ""
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        page_text = self._formatTables(tables)
                    
                    # Extract text if no tables
                    if not page_text:
                        page_text = page.extract_text() or ""
                    
                    extracted_text.append(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber processing error: {e}")
        
        return extracted_text, page_count
    
    def _shouldAttemptOCR(self, text_pages: List[str]) -> bool:
        """Determine if OCR should be attempted"""
        if not text_pages:
            return True
        
        total_text = ''.join(text_pages)
        
        # Check if text is too short or mostly whitespace
        if len(total_text.strip()) < 100:
            return True
        
        # Check if text has too many non-printable characters
        non_printable_ratio = sum(1 for c in total_text if ord(c) < 32) / max(len(total_text), 1)
        if non_printable_ratio > 0.1:
            return True
        
        return False
    
    def _performOCRWithValidation(self, pdf_source) -> Tuple[List[str], int]:
        """Perform OCR with validation and return text with page count"""
        if not self.ocr_available:
            logger.warning("OCR skipped - Tesseract not available")
            return [], 0
        
        ocr_text = []
        page_count = 0
        
        try:
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_bytes, convert_from_path
            
            # Convert PDF to images
            if isinstance(pdf_source, io.BytesIO):
                pdf_source.seek(0)
                images = convert_from_bytes(pdf_source.read())
            else:
                images = convert_from_path(pdf_source)
            
            if not images:
                raise ProcessingError("No images could be extracted from PDF for OCR")
            
            page_count = len(images)
            
            # Process each page
            for pageNum, image in enumerate(images):
                try:
                    # Perform OCR
                    text = pytesseract.image_to_string(image, lang='eng')
                    
                    # Validate OCR output
                    if text and len(text.strip()) > 10:
                        ocr_text.append(text)
                        logger.info(f"OCR successful for page {pageNum + 1}")
                    else:
                        ocr_text.append("")
                        logger.warning(f"OCR produced no meaningful text for page {pageNum + 1}")
                        
                except Exception as e:
                    logger.error(f"OCR failed for page {pageNum + 1}: {e}")
                    ocr_text.append("")
            
            # Check if OCR was successful overall
            if not any(text for text in ocr_text):
                raise ProcessingError("OCR failed to extract any meaningful text")
            
            return ocr_text, page_count
            
        except ImportError as e:
            raise ProcessingError(f"OCR dependencies not installed: {e}")
        except Exception as e:
            raise ProcessingError(f"OCR processing failed: {e}")
    
    def _extractHeaders(self, text: str) -> List[str]:
        """Extract headers and section titles from text"""
        if not text:
            return []
        
        headers = []
        
        # Common header patterns
        patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][^.!?]*):$',  # Title case with colon
            r'^(\d+\.?\d*\.?\s+[A-Z].+)$',  # Numbered sections
            r'^([A-Z\s]+)$',  # All caps lines (max 50 chars)
        ]
        
        lines = text.split('\n')
        for line in lines[:100]:  # Check first 100 lines only
            line = line.strip()
            if 5 < len(line) < 100:  # Reasonable header length
                for pattern in patterns:
                    if match := re.match(pattern, line):
                        header = match.group(1).strip()
                        if header and header not in headers:
                            headers.append(header)
                        break
        
        return headers[:10]  # Limit to 10 headers
    
    def _enrichTextWithHeaders(self, text: str, headers: List[str]) -> str:
        """Add header context to improve retrieval"""
        if not headers or not text:
            return text
        
        headerContext = f"Section Headers: {', '.join(headers[:3])}\n\n"
        return headerContext + text
    
    def _formatTables(self, tables: List[List[List[str]]]) -> str:
        """Format extracted tables as readable text"""
        if not tables:
            return ""
        
        formattedTables = []
        
        for tableIdx, table in enumerate(tables[:5]):  # Limit to 5 tables
            if table:
                tableText = f"Table {tableIdx + 1}:\n"
                for row in table[:50]:  # Limit rows
                    if row:
                        tableText += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                formattedTables.append(tableText)
        
        return "\n".join(formattedTables)
    
    def _createChunksWithValidation(self, extracted_content: Dict[str, Any], 
                                   document_id: str, file_path: str) -> List[DocumentChunk]:
        """Create chunks with validation and accurate page tracking"""
        chunks = []
        
        # Get the accurate page count
        total_pages = extracted_content['metadata'].get('page_count', 0)
        
        # Process text pages into chunks
        all_text = extracted_content.get('text', [])
        if isinstance(all_text, list):
            for pageNum, pageText in enumerate(all_text):
                if not pageText or len(pageText.strip()) < 10:
                    continue
                
                try:
                    # Use semantic chunking if enabled
                    if self.config.semantic_chunking and self.semantic_chunker:
                        pageChunks = self.semantic_chunker.extractSemanticSegments(pageText)
                    else:
                        pageChunks = self._simpleChunking(pageText)
                    
                    # Create DocumentChunk objects
                    for chunkIdx, chunkText in enumerate(pageChunks):
                        if chunkText and len(chunkText.strip()) > 10:
                            try:
                                # Get page data if available
                                pages = extracted_content.get('pages', [])
                                page_data = pages[pageNum] if pageNum < len(pages) else {}
                                
                                chunk = DocumentChunk(
                                    chunk_id=f"{document_id}_{pageNum}_{chunkIdx}",
                                    document_id=document_id,
                                    content=chunkText,
                                    page_number=pageNum + 1,
                                    chunk_index=chunkIdx,
                                    metadata={
                                        'file_path': file_path,
                                        'page_count': total_pages,  # Use accurate page count
                                        'headers': page_data.get('headers', []),
                                        'has_tables': 'Table' in chunkText,
                                        'extraction_method': extracted_content.get('extraction_method', 'unknown')
                                    }
                                )
                                
                                if chunk.is_valid():
                                    chunks.append(chunk)
                            except ValueError as e:
                                logger.warning(f"Invalid chunk skipped: {e}")
                                
                except Exception as e:
                    logger.error(f"Error creating chunks for page {pageNum + 1}: {e}")
        
        return chunks
    
    def _simpleChunking(self, text: str) -> List[str]:
        """Simple chunking with overlap"""
        if not text:
            return []
        
        chunks = []
        textLength = len(text)
        
        # Adjust chunk size based on document length
        chunk_size = self.config.getOptimalChunkSize(textLength)
        chunk_overlap = min(self.config.chunk_overlap, chunk_size // 2)
        
        start = 0
        while start < textLength:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < textLength:
                # Look for sentence end
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    sentenceEnd = text.rfind(delimiter, start, end)
                    if sentenceEnd != -1:
                        end = sentenceEnd + len(delimiter)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move with overlap
            start = end - chunk_overlap if chunk_overlap > 0 else end
        
        return chunks
    
    def processBatch(self, file_paths: List[str]) -> Dict[str, ProcessedDocument]:
        """Process multiple PDFs with error collection"""
        results = {}
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.processPDF, path): path 
                for path in file_paths
            }
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    processed_doc = future.result()
                    results[path] = processed_doc
                    logger.info(f"Successfully processed: {path} ({processed_doc.page_count} pages)")
                except Exception as e:
                    error_msg = f"Failed to process {path}: {e}"
                    logger.error(error_msg)
                    errors.append({'file': path, 'error': str(e)})
                    results[path] = None
        
        if errors:
            logger.warning(f"Batch processing completed with {len(errors)} errors")
        
        return results
    
    def getProcessingErrors(self) -> List[Dict]:
        """Get list of processing errors"""
        return self.processing_errors
    
    def clearCache(self):
        """Clear processed documents cache"""
        self.processed_documents.clear()
        logger.info("Document cache cleared")
    
    def getDocumentInfo(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document"""
        if document_id in self.processed_documents:
            doc = self.processed_documents[document_id]
            return doc.to_dict()
        return None

# Factory pattern for processor instantiation
def createPDFProcessor() -> PDFProcessor:
    """Factory method to create PDF processor"""
    return PDFProcessor()