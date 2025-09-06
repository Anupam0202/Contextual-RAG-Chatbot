"""
Vector Store Abstraction Layer
Supports multiple backends: in-memory, FAISS, Pinecone
"""

import abc
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import hashlib

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import getGlobalConfig, getSecurityConfig
from pdf_processor import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    chunk: DocumentChunk
    score: float
    retrieval_method: str  # 'dense', 'sparse', 'hybrid'
    metadata: Dict[str, Any]

class VectorStoreInterface(abc.ABC):
    """Abstract interface for vector stores"""
    
    @abc.abstractmethod
    def addDocuments(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the store"""
        pass
    
    @abc.abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abc.abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete a document from the store"""
        pass
    
    @abc.abstractmethod
    def save(self, path: str) -> bool:
        """Persist the vector store"""
        pass
    
    @abc.abstractmethod
    def load(self, path: str) -> bool:
        """Load the vector store"""
        pass

class InMemoryVectorStore(VectorStoreInterface):
    """In-memory vector store with hybrid search and auto-persistence"""
    
    def __init__(self):
        self.config = getGlobalConfig()
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        self.chunks = []
        self.embeddings = []
        self.bm25_index = None
        self.document_map = {}
        
        # Auto-persistence
        self.persistence_path = Path("data/vector_store.pkl")
        self.persistence_path.parent.mkdir(exist_ok=True)
        
        # Auto-load if exists
        if self.persistence_path.exists():
            try:
                self.load(str(self.persistence_path))
                logger.info(f"Loaded existing vector store from {self.persistence_path}")
            except Exception as e:
                logger.warning(f"Could not load existing vector store: {e}")
        
    def addDocuments(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks with embeddings and auto-persistence"""
        try:
            for chunk in chunks:
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.content)
                chunk.embedding = embedding
                
                # Store
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                
                # Update document map
                if chunk.document_id not in self.document_map:
                    self.document_map[chunk.document_id] = []
                self.document_map[chunk.document_id].append(len(self.chunks) - 1)
            
            # Rebuild BM25 index
            self._rebuildBM25Index()
            
            # Auto-save
            self.save(str(self.persistence_path))
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def _rebuildBM25Index(self):
        """Rebuild BM25 sparse index"""
        if self.chunks:
            tokenizedChunks = [chunk.content.lower().split() for chunk in self.chunks]
            self.bm25_index = BM25Okapi(tokenizedChunks)
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Hybrid search combining dense and sparse retrieval"""
        if not self.chunks:
            return []
        
        # Dense retrieval (semantic)
        denseResults = self._denseSearch(query, top_k * 2)
        
        # Sparse retrieval (keyword)
        sparseResults = self._sparseSearch(query, top_k * 2)
        
        # Combine results
        hybridResults = self._fuseResults(denseResults, sparseResults, top_k)
        
        return hybridResults
    
    def _denseSearch(self, query: str, top_k: int) -> List[SearchResult]:
        """Semantic search using embeddings"""
        queryEmbedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            similarity = np.dot(queryEmbedding, embedding) / (
                np.linalg.norm(queryEmbedding) * np.linalg.norm(embedding) + 1e-10
            )
            similarities.append((idx, similarity))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        topResults = similarities[:min(top_k, len(similarities))]
        
        results = []
        for idx, score in topResults:
            if score >= self.config.similarity_threshold:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=score,
                    retrieval_method='dense',
                    metadata={'similarity_type': 'cosine'}
                ))
        
        return results
    
    def _sparseSearch(self, query: str, top_k: int) -> List[SearchResult]:
        """Keyword search using BM25"""
        if not self.bm25_index:
            return []
        
        tokenizedQuery = query.lower().split()
        scores = self.bm25_index.get_scores(tokenizedQuery)
        
        # Get top-k indices
        topIndices = np.argsort(scores)[-min(top_k, len(scores)):][::-1]
        
        results = []
        for idx in topIndices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    retrieval_method='sparse',
                    metadata={'algorithm': 'BM25'}
                ))
        
        return results
    
    def _fuseResults(self, dense_results: List[SearchResult], 
                     sparse_results: List[SearchResult], 
                     top_k: int) -> List[SearchResult]:
        """Fuse dense and sparse results using weighted combination"""
        # Create score dictionaries
        denseScores = {r.chunk.chunk_id: r.score for r in dense_results}
        sparseScores = {r.chunk.chunk_id: r.score for r in sparse_results}
        
        # Normalize scores
        maxDense = max(denseScores.values()) if denseScores else 1
        maxSparse = max(sparseScores.values()) if sparseScores else 1
        
        # Combine all unique chunks
        allChunkIds = set(denseScores.keys()) | set(sparseScores.keys())
        
        fusedResults = []
        for chunkId in allChunkIds:
            # Get normalized scores
            denseScore = denseScores.get(chunkId, 0) / (maxDense + 1e-10)
            sparseScore = sparseScores.get(chunkId, 0) / (maxSparse + 1e-10)
            
            # Weighted combination
            hybridScore = (
                self.config.hybrid_search_alpha * denseScore +
                (1 - self.config.hybrid_search_alpha) * sparseScore
            )
            
            # Find the chunk
            chunk = next((r.chunk for r in dense_results + sparse_results 
                         if r.chunk.chunk_id == chunkId), None)
            
            if chunk:
                fusedResults.append(SearchResult(
                    chunk=chunk,
                    score=hybridScore,
                    retrieval_method='hybrid',
                    metadata={
                        'dense_score': denseScore,
                        'sparse_score': sparseScore
                    }
                ))
        
        # Sort by hybrid score and return top-k
        fusedResults.sort(key=lambda x: x.score, reverse=True)
        return fusedResults[:top_k]
    
    def delete(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        if document_id not in self.document_map:
            return False
        
        # Get indices to delete (in reverse order to maintain indices)
        indices = sorted(self.document_map[document_id], reverse=True)
        
        for idx in indices:
            del self.chunks[idx]
            del self.embeddings[idx]
        
        # Remove from document map
        del self.document_map[document_id]
        
        # Rebuild BM25 index
        self._rebuildBM25Index()
        
        # Auto-save
        self.save(str(self.persistence_path))
        
        return True
    
    def save(self, path: str) -> bool:
        """Save vector store to disk"""
        try:
            saveData = {
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'document_map': self.document_map
            }
            
            with open(path, 'wb') as f:
                pickle.dump(saveData, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load vector store from disk"""
        try:
            with open(path, 'rb') as f:
                saveData = pickle.load(f)
            
            self.chunks = saveData.get('chunks', [])
            self.embeddings = saveData.get('embeddings', [])
            self.document_map = saveData.get('document_map', {})
            
            # Rebuild BM25 index
            self._rebuildBM25Index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

class FAISSVectorStore(VectorStoreInterface):
    """FAISS-based vector store for production use"""
    
    def __init__(self):
        self.config = getGlobalConfig()
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Initialize FAISS index
        self.dimension = self.config.vector_dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add IVF for large-scale indexing
        if self.config.vector_store_type == 'faiss_ivf':
            nlist = 100
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        self.chunks = []
        self.bm25_index = None
        
        # Auto-persistence
        self.persistence_path = Path("data/faiss_store")
        self.persistence_path.mkdir(exist_ok=True)
        
        # Auto-load if exists
        if (self.persistence_path / "index.faiss").exists():
            try:
                self.load(str(self.persistence_path))
                logger.info(f"Loaded existing FAISS store from {self.persistence_path}")
            except Exception as e:
                logger.warning(f"Could not load existing FAISS store: {e}")
        
    def addDocuments(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to FAISS index"""
        try:
            embeddings = []
            
            for chunk in chunks:
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.content)
                
                # Ensure correct dimension
                if len(embedding) != self.dimension:
                    embedding = np.resize(embedding, self.dimension)
                
                embeddings.append(embedding)
                chunk.embedding = embedding
                self.chunks.append(chunk)
            
            # Add to FAISS
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Train if using IVF
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # Rebuild BM25
            self._rebuildBM25Index()
            
            # Auto-save
            self.save(str(self.persistence_path))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to FAISS: {e}")
            return False
    
    def _rebuildBM25Index(self):
        """Rebuild BM25 index"""
        if self.chunks:
            tokenizedChunks = [chunk.content.lower().split() for chunk in self.chunks]
            self.bm25_index = BM25Okapi(tokenizedChunks)
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search using FAISS and BM25"""
        if not self.chunks:
            return []
        
        # Dense search with FAISS
        queryEmbedding = self.embedding_model.encode(query)
        if len(queryEmbedding) != self.dimension:
            queryEmbedding = np.resize(queryEmbedding, self.dimension)
        
        queryArray = np.array([queryEmbedding]).astype('float32')
        distances, indices = self.index.search(queryArray, min(top_k * 2, len(self.chunks)))
        
        denseResults = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + distance)
                denseResults.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=similarity,
                    retrieval_method='dense',
                    metadata={'faiss_distance': float(distance)}
                ))
        
        # Sparse search with BM25
        sparseResults = self._sparseSearch(query, top_k * 2)
        
        # Fuse results
        return self._fuseResults(denseResults, sparseResults, top_k)
    
    def _sparseSearch(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 search"""
        if not self.bm25_index:
            return []
        
        tokenizedQuery = query.lower().split()
        scores = self.bm25_index.get_scores(tokenizedQuery)
        topIndices = np.argsort(scores)[-min(top_k, len(scores)):][::-1]
        
        results = []
        for idx in topIndices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    retrieval_method='sparse',
                    metadata={'algorithm': 'BM25'}
                ))
        
        return results
    
    def _fuseResults(self, dense_results: List[SearchResult], 
                     sparse_results: List[SearchResult], 
                     top_k: int) -> List[SearchResult]:
        """Reciprocal Rank Fusion"""
        # RRF scoring
        k = 60  # RRF constant
        
        scores = {}
        
        # Score dense results
        for rank, result in enumerate(dense_results):
            chunkId = result.chunk.chunk_id
            scores[chunkId] = scores.get(chunkId, 0) + 1 / (k + rank + 1)
        
        # Score sparse results
        for rank, result in enumerate(sparse_results):
            chunkId = result.chunk.chunk_id
            scores[chunkId] = scores.get(chunkId, 0) + 1 / (k + rank + 1)
        
        # Create fused results
        fusedResults = []
        chunkMap = {r.chunk.chunk_id: r.chunk for r in dense_results + sparse_results}
        
        for chunkId, score in scores.items():
            if chunkId in chunkMap:
                fusedResults.append(SearchResult(
                    chunk=chunkMap[chunkId],
                    score=score,
                    retrieval_method='hybrid',
                    metadata={'fusion_method': 'RRF'}
                ))
        
        fusedResults.sort(key=lambda x: x.score, reverse=True)
        return fusedResults[:top_k]
    
    def delete(self, document_id: str) -> bool:
        """Remove document from FAISS"""
        # FAISS doesn't support direct deletion, need to rebuild
        newChunks = [c for c in self.chunks if c.document_id != document_id]
        
        if len(newChunks) == len(self.chunks):
            return False
        
        # Rebuild index
        self.chunks = []
        self.index = faiss.IndexFlatL2(self.dimension)
        
        if newChunks:
            self.addDocuments(newChunks)
        
        return True
    
    def save(self, path: str) -> bool:
        """Save FAISS index and chunks"""
        try:
            path = Path(path)
            path.mkdir(exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(path / "index.faiss"))
            
            # Save chunks
            with open(path / "chunks.pkl", 'wb') as f:
                pickle.dump(self.chunks, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS store: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load FAISS index and chunks"""
        try:
            path = Path(path)
            
            # Load FAISS index
            self.index = faiss.read_index(str(path / "index.faiss"))
            
            # Load chunks
            with open(path / "chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            # Rebuild BM25
            self._rebuildBM25Index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS store: {e}")
            return False

# Factory pattern for vector store creation
class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def createVectorStore(store_type: Optional[str] = None) -> VectorStoreInterface:
        """Create appropriate vector store based on configuration"""
        
        if store_type is None:
            store_type = getGlobalConfig().vector_store_type
        
        if store_type == 'faiss' or store_type == 'faiss_ivf':
            return FAISSVectorStore()
        else:
            return InMemoryVectorStore()

# Singleton pattern for global vector store
_globalVectorStore = None

def getGlobalVectorStore() -> VectorStoreInterface:
    """Get or create global vector store instance"""
    global _globalVectorStore
    
    if _globalVectorStore is None:
        _globalVectorStore = VectorStoreFactory.createVectorStore()
    
    return _globalVectorStore