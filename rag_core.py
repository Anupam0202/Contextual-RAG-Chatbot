"""
Core RAG Engine with Advanced Features
Implements planning, reflection, and tool use patterns
"""

import asyncio
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import os

import google.generativeai as genai

from config import getGlobalConfig, getSecurityConfig
from vector_store import getGlobalVectorStore, SearchResult
from pdf_processor import DocumentChunk

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent classification"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    CLARIFICATION = "clarification"

@dataclass
class QueryPlan:
    """Query execution plan"""
    original_query: str
    intent: QueryIntent
    sub_queries: List[str]
    required_context: List[str]
    search_strategy: str  # 'dense', 'sparse', 'hybrid'
    reasoning_depth: int  # 1-5 scale
    
@dataclass
class GenerationContext:
    """Context for generation"""
    query: str
    retrieved_chunks: List[SearchResult]
    plan: QueryPlan
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]

class ThinkingRAG:
    """RAG with thinking/reasoning capabilities and proper context handling"""
    
    def __init__(self):
        self.config = getGlobalConfig()
        self.security = getSecurityConfig()
        self.vector_store = getGlobalVectorStore()
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.model_name)
        
        # Safety settings
        self.safety_settings = {
            'HATE': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
            'SEXUAL': 'BLOCK_MEDIUM_AND_ABOVE',
            'DANGEROUS': 'BLOCK_MEDIUM_AND_ABOVE'
        }
        
        self.conversation_memory = []
        self.query_cache = {}
        
    async def processQuery(self, query: str, 
                          conversation_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        """Process query with planning, retrieval, and generation - FIXED context handling"""
        
        try:
            # Check cache
            cacheKey = hashlib.md5(query.encode()).hexdigest()
            if self.config.enable_caching and cacheKey in self.query_cache:
                cached = self.query_cache[cacheKey]
                if datetime.now() - cached['timestamp'] < timedelta(seconds=self.config.cache_ttl):
                    yield cached['response']
                    return
            
            # Step 1: Planning
            plan = await self._planQuery(query, conversation_history)
            
            # Step 2: Retrieval
            retrieved_chunks = await self._executeRetrieval(plan)
            
            # Step 3: Generation with streaming
            context = GenerationContext(
                query=query,
                retrieved_chunks=retrieved_chunks,
                plan=plan,
                conversation_history=conversation_history or [],
                metadata={'timestamp': datetime.now().isoformat()}
            )
            
            fullResponse = ""
            async for chunk in self._generateResponse(context):
                fullResponse += chunk
                yield chunk
            
            # Step 4: Reflection and improvement (optional)
            if plan.reasoning_depth >= 3:
                improvedResponse = await self._reflectAndImprove(fullResponse, context)
                if improvedResponse != fullResponse:
                    yield "\n\n**[Refined Response]**\n"
                    yield improvedResponse
                    fullResponse = improvedResponse
            
            # Cache response
            if self.config.enable_caching:
                self.query_cache[cacheKey] = {
                    'response': fullResponse,
                    'timestamp': datetime.now()
                }
            
            # Update conversation memory
            self.conversation_memory.append({
                'query': query,
                'response': fullResponse,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield f"I apologize, but I encountered an error processing your query: {str(e)}"
    
    async def _planQuery(self, query: str, conversation_history: Optional[List[Dict[str, str]]]) -> QueryPlan:
        """Create execution plan for query"""
        
        # Classify intent
        intent = await self._classifyIntent(query)
        
        # Decompose query if complex
        sub_queries = await self._decomposeQuery(query, intent)
        
        # Determine required context
        required_context = self._identifyRequiredContext(query, intent)
        
        # Choose search strategy
        search_strategy = self._selectSearchStrategy(intent, query)
        
        # Determine reasoning depth
        reasoning_depth = self._assessReasoningDepth(query, intent)
        
        return QueryPlan(
            original_query=query,
            intent=intent,
            sub_queries=sub_queries,
            required_context=required_context,
            search_strategy=search_strategy,
            reasoning_depth=reasoning_depth
        )
    
    async def _classifyIntent(self, query: str) -> QueryIntent:
        """Classify query intent using Gemini"""
        
        prompt = f"""Classify the following query into one of these categories:
        - FACTUAL: Seeking specific facts or information
        - ANALYTICAL: Requiring analysis or comparison
        - CREATIVE: Requesting creative output
        - CONVERSATIONAL: General conversation
        - TECHNICAL: Technical or code-related
        - CLARIFICATION: Seeking clarification
        
        Query: {query}
        
        Return only the category name."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=20
                )
            )
            
            intentStr = response.text.strip().upper()
            
            # Map to enum value
            for intent in QueryIntent:
                if intent.name == intentStr:
                    return intent
            
            return QueryIntent.FACTUAL
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return QueryIntent.FACTUAL
    
    async def _decomposeQuery(self, query: str, intent: QueryIntent) -> List[str]:
        """Decompose complex query into sub-queries"""
        
        if intent not in [QueryIntent.ANALYTICAL, QueryIntent.TECHNICAL]:
            return [query]
        
        prompt = f"""Decompose this query into simpler sub-questions that need to be answered:
        Query: {query}
        
        Return a JSON list of sub-questions. If the query is simple, return just the original query.
        Example: ["What is X?", "How does Y work?", "What is the relationship between X and Y?"]"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200
                )
            )
            
            # Extract JSON from response
            jsonMatch = re.search(r'```math.*?```', response.text, re.DOTALL)
            if jsonMatch:
                sub_queries = json.loads(jsonMatch.group())
                return sub_queries if isinstance(sub_queries, list) else [query]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
        
        return [query]
    
    def _identifyRequiredContext(self, query: str, intent: QueryIntent) -> List[str]:
        """Identify required context types"""
        
        contexts = []
        
        # Check for temporal context
        if any(word in query.lower() for word in ['when', 'date', 'year', 'time', 'recent', 'latest']):
            contexts.append('temporal')
        
        # Check for comparison context
        if any(word in query.lower() for word in ['compare', 'difference', 'vs', 'versus', 'better']):
            contexts.append('comparative')
        
        # Check for causal context
        if any(word in query.lower() for word in ['why', 'because', 'cause', 'reason', 'how']):
            contexts.append('causal')
        
        # Intent-based context
        if intent == QueryIntent.TECHNICAL:
            contexts.extend(['technical', 'implementation'])
        elif intent == QueryIntent.ANALYTICAL:
            contexts.extend(['analytical', 'statistical'])
        
        return contexts
    
    def _selectSearchStrategy(self, intent: QueryIntent, query: str) -> str:
        """Select optimal search strategy"""
        
        # Technical queries benefit from keyword search
        if intent == QueryIntent.TECHNICAL:
            return 'hybrid'
        
        # Factual queries with specific terms
        if intent == QueryIntent.FACTUAL and len(query.split()) <= 5:
            return 'sparse'
        
        # Complex analytical queries need semantic understanding
        if intent == QueryIntent.ANALYTICAL:
            return 'dense'
        
        # Default to hybrid for best results
        return 'hybrid'
    
    def _assessReasoningDepth(self, query: str, intent: QueryIntent) -> int:
        """Assess required reasoning depth (1-5)"""
        
        depth = 1
        
        # Increase for analytical queries
        if intent == QueryIntent.ANALYTICAL:
            depth += 2
        
        # Increase for technical queries
        if intent == QueryIntent.TECHNICAL:
            depth += 1
        
        # Increase for complex language patterns
        complexPatterns = ['explain', 'analyze', 'compare', 'evaluate', 'assess']
        if any(pattern in query.lower() for pattern in complexPatterns):
            depth += 1
        
        # Cap at 5
        return min(depth, 5)
    
    async def _executeRetrieval(self, plan: QueryPlan) -> List[SearchResult]:
        """Execute retrieval based on plan"""
        
        allResults = []
        
        # Retrieve for each sub-query
        for sub_query in plan.sub_queries:
            # Apply search strategy
            results = self.vector_store.search(sub_query, self.config.retrieval_top_k)
            
            # Filter by strategy if needed
            if plan.search_strategy == 'dense':
                results = [r for r in results if r.retrieval_method in ['dense', 'hybrid']]
            elif plan.search_strategy == 'sparse':
                results = [r for r in results if r.retrieval_method in ['sparse', 'hybrid']]
            
            allResults.extend(results)
        
        # Deduplicate and rerank
        uniqueResults = self._deduplicateResults(allResults)
        
        # Rerank if enabled
        if self.config.rerank_enabled and uniqueResults:
            uniqueResults = await self._rerankResults(uniqueResults, plan.original_query)
        
        return uniqueResults[:self.config.retrieval_top_k]
    
    def _deduplicateResults(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results"""
        
        seen = set()
        unique = []
        
        for result in results:
            if result.chunk.chunk_id not in seen:
                seen.add(result.chunk.chunk_id)
                unique.append(result)
        
        return unique
    
    async def _rerankResults(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results using Gemini"""
        
        if not results:
            return results
        
        # Create reranking prompt
        chunks_text = "\n\n".join([
            f"[{i+1}] {r.chunk.content[:500]}..." 
            for i, r in enumerate(results[:10])  # Limit to top 10 for reranking
        ])
        
        prompt = f"""Given the query and retrieved passages, rank them by relevance.
        Query: {query}
        
        Passages:
        {chunks_text}
        
        Return a JSON list of passage numbers in order of relevance.
        Example: [3, 1, 5, 2, 4]"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=50
                )
            )
            
            # Parse ranking
            jsonMatch = re.search(r'```math.*?```', response.text)
            if jsonMatch:
                ranking = json.loads(jsonMatch.group())
                
                # Reorder results
                reranked = []
                for idx in ranking:
                    if 1 <= idx <= len(results):
                        reranked.append(results[idx - 1])
                
                # Add any missing results
                for result in results:
                    if result not in reranked:
                        reranked.append(result)
                
                return reranked
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        
        return results
    
    async def _generateResponse(self, context: GenerationContext) -> AsyncGenerator[str, None]:
        """Generate response with streaming"""
        
        # Build context prompt with proper conversation history
        contextPrompt = self._buildContextPrompt(context)
        
        try:
            # Generate response
            response = self.model.generate_content(
                contextPrompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                ),
                safety_settings=self.safety_settings,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            yield f"Error generating response: {str(e)}"
    
    def _buildContextPrompt(self, context: GenerationContext) -> str:
        """Build prompt with retrieved context and conversation history - FIXED"""
        
        # Format retrieved chunks
        contextText = "\n\n".join([
            f"[Source {i+1} - Page {r.chunk.page_number}]:\n{r.chunk.content}"
            for i, r in enumerate(context.retrieved_chunks)
        ])
        
        # Build conversation context WITH PROPER FORMATTING
        conversationContext = ""
        if context.conversation_history:
            conversationContext = "Previous conversation context:\n"
            
            # Format conversation history properly
            for msg in context.conversation_history:
                if msg.get('role') == 'user':
                    conversationContext += f"\nUser: {msg.get('content', '')}"
                elif msg.get('role') == 'assistant':
                    conversationContext += f"\nAssistant: {msg.get('content', '')}\n"
        
        # Intent-specific instructions
        intentInstructions = self._getIntentInstructions(context.plan.intent)
        
        # Construct full prompt with clear separation
        prompt = f"""{intentInstructions}

{conversationContext}

Retrieved Information:
{contextText}

Current Query: {context.query}

Instructions:
1. Use the conversation context to understand the flow of discussion
2. Reference retrieved information with [Source N] citations
3. Maintain consistency with previous responses
4. If context is insufficient, acknowledge this clearly
5. Provide a comprehensive, well-structured response
"""
        
        return prompt
    
    def _getIntentInstructions(self, intent: QueryIntent) -> str:
        """Get intent-specific instructions"""
        
        instructions = {
            QueryIntent.FACTUAL: "Provide accurate, factual information with clear citations.",
            QueryIntent.ANALYTICAL: "Analyze the information thoroughly, comparing different aspects and drawing insights.",
            QueryIntent.TECHNICAL: "Provide detailed technical explanation with code examples if relevant.",
            QueryIntent.CREATIVE: "Use the context creatively while maintaining accuracy.",
            QueryIntent.CONVERSATIONAL: "Respond naturally and conversationally while being informative.",
            QueryIntent.CLARIFICATION: "Clarify the concepts clearly with examples."
        }
        
        return instructions.get(intent, "Provide a helpful and accurate response.")
    
    async def _reflectAndImprove(self, response: str, context: GenerationContext) -> str:
        """Reflect on response and improve if needed"""
        
        reflectionPrompt = f"""Review this response for accuracy, completeness, and clarity:

Query: {context.query}

Response: {response}

Identify any issues:
1. Are all parts of the query addressed?
2. Are citations properly included?
3. Is the information accurate based on the context?
4. Is the response clear and well-structured?

If improvements are needed, provide an improved version. Otherwise, return "NO_CHANGES_NEEDED"."""
        
        try:
            reflectionResponse = self.model.generate_content(
                reflectionPrompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=self.config.max_output_tokens
                )
            )
            
            if "NO_CHANGES_NEEDED" in reflectionResponse.text:
                return response
            
            return reflectionResponse.text
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return response

class AgenticRAG(ThinkingRAG):
    """Extended RAG with agentic capabilities"""
    
    def __init__(self):
        super().__init__()
    
    async def processWithTools(self, query: str) -> AsyncGenerator[str, None]:
        """Process query with tool use capability"""
        
        # Determine if tools are needed
        needsTools = await self._assessToolNeed(query)
        
        if needsTools:
            # Execute with tools
            async for chunk in self._executeWithTools(query):
                yield chunk
        else:
            # Standard processing
            async for chunk in self.processQuery(query):
                yield chunk
    
    async def _assessToolNeed(self, query: str) -> bool:
        """Assess if query needs tool execution"""
        
        toolIndicators = ['calculate', 'compute', 'code', 'implement', 'algorithm', 'function']
        return any(indicator in query.lower() for indicator in toolIndicators)
    
    async def _executeWithTools(self, query: str) -> AsyncGenerator[str, None]:
        """Execute query with tool support"""
        
        # First, get context
        plan = await self._planQuery(query, None)
        retrieved = await self._executeRetrieval(plan)
        
        # Build tool-aware prompt
        contextText = "\n\n".join([r.chunk.content for r in retrieved[:3]])
        
        toolPrompt = f"""You have access to code execution capabilities.
        
Context: {contextText}

Query: {query}

Provide a response with code examples where helpful. Show your work."""
        
        try:
            # Generate with tools
            response = self.model.generate_content(
                toolPrompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens
                ),
                safety_settings=self.safety_settings,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error with tool execution: {str(e)}"

# Global RAG instance management
_globalRAGInstance = None

def getRAGEngine() -> ThinkingRAG:
    """Get or create global RAG engine"""
    global _globalRAGInstance
    
    if _globalRAGInstance is None:
        _globalRAGInstance = AgenticRAG()
    
    return _globalRAGInstance