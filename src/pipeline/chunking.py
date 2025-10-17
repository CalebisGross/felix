"""
Output Chunking and Streaming for Felix Framework.

This module introduces data structures and logic for intelligent output chunking
and streaming, enabling progressive refinement and efficient handling of large
language model responses.

Key Features:
- ChunkedResult: Represents a single chunk of an LLM response.
- ProgressiveProcessor: Manages the incremental generation and processing of chunks.
- ContentSummarizer: Provides smart truncation and summary fallback for content.
"""

import uuid
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class ChunkedResult:
    """Represents a single chunk of an LLM response."""
    chunk_id: str
    task_id: str
    agent_id: str
    content_chunk: str
    chunk_index: int
    is_final: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    continuation_token: Optional[str] = None # Token to request next chunk

@dataclass
class ProgressiveProcessor:
    """
    Manages the incremental generation and processing of chunks.
    
    This class helps an agent break down its output into manageable chunks
    and provides mechanisms for other agents to request subsequent chunks.
    """
    task_id: str
    agent_id: str
    full_content: str
    chunk_size: int = 500 # Default chunk size in characters
    _current_chunk_index: int = 0
    _continuation_tokens: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        self._total_chunks = max(1, (len(self.full_content) + self.chunk_size - 1) // self.chunk_size)
        self._generate_continuation_tokens()

    def _generate_continuation_tokens(self):
        """Generates unique continuation tokens for each chunk."""
        for i in range(self._total_chunks):
            self._continuation_tokens[i] = str(uuid.uuid4())

    def get_next_chunk(self, requested_token: Optional[str] = None) -> Optional[ChunkedResult]:
        """
        Retrieves the next chunk of content.
        
        Args:
            requested_token: The continuation token from the previous chunk,
                             or None for the first chunk.
        
        Returns:
            A ChunkedResult object, or None if no more chunks.
        """
        if requested_token is None:
            # Requesting the first chunk
            current_index = 0
        else:
            # Find the index corresponding to the requested token
            found_index = -1
            for idx, token in self._continuation_tokens.items():
                if token == requested_token:
                    found_index = idx  # Token points to its own chunk index
                    break
            
            if found_index == -1 or found_index >= self._total_chunks:
                return None # Invalid token or no more chunks
            current_index = found_index

        start_idx = current_index * self.chunk_size
        end_idx = min((current_index + 1) * self.chunk_size, len(self.full_content))
        
        if current_index >= self._total_chunks:
            return None # No more chunks

        content_chunk = self.full_content[start_idx:end_idx]
        is_final = (current_index == self._total_chunks - 1)
        
        continuation_token = None
        if not is_final:
            continuation_token = self._continuation_tokens.get(current_index + 1)

        return ChunkedResult(
            chunk_id=str(uuid.uuid4()),
            task_id=self.task_id,
            agent_id=self.agent_id,
            content_chunk=content_chunk,
            chunk_index=current_index,
            is_final=is_final,
            timestamp=time.time(),
            continuation_token=continuation_token
        )

    def get_chunk_by_index(self, index: int) -> Optional[ChunkedResult]:
        """
        Retrieves a specific chunk by its index.
        
        Args:
            index: The 0-based index of the chunk to retrieve.
            
        Returns:
            A ChunkedResult object, or None if the index is out of bounds.
        """
        if not (0 <= index < self._total_chunks):
            return None

        start_idx = index * self.chunk_size
        end_idx = min((index + 1) * self.chunk_size, len(self.full_content))
        
        content_chunk = self.full_content[start_idx:end_idx]
        is_final = (index == self._total_chunks - 1)
        
        continuation_token = None
        if not is_final:
            continuation_token = self._continuation_tokens.get(index + 1)

        return ChunkedResult(
            chunk_id=str(uuid.uuid4()),
            task_id=self.task_id,
            agent_id=self.agent_id,
            content_chunk=content_chunk,
            chunk_index=index,
            is_final=is_final,
            timestamp=time.time(),
            continuation_token=continuation_token
        )

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

class ContentSummarizer:
    """
    Provides smart truncation and summary fallback for content.
    
    This class can be used to generate summaries of content,
    especially when the original content exceeds certain limits.
    """
    
    def __init__(self, llm_client: Any): # Use Any to avoid circular dependency for now
        """
        Initialize ContentSummarizer.
        
        Args:
            llm_client: An LLM client (e.g., LMStudioClient) to generate summaries.
        """
        self.llm_client = llm_client

    async def summarize_content(self, content: str, target_tokens: int, 
                                agent_id: str, task_id: str) -> str:
        """
        Generates a summary of the given content to fit within target_tokens.
        
        Args:
            content: The full content to summarize.
            target_tokens: The maximum number of tokens for the summary.
            agent_id: The ID of the agent requesting the summary.
            task_id: The ID of the task associated with the content.
            
        Returns:
            The summarized content.
        """
        if not content:
            return ""

        # Estimate current token count (simple approximation)
        current_tokens = len(content.split()) * 1.3 # Rough word to token ratio

        if current_tokens <= target_tokens:
            return content # No summarization needed

        # Create a prompt for summarization
        system_prompt = f"""You are a Content Summarizer. Your goal is to condense the provided text into a concise summary that captures all essential information, without exceeding {target_tokens} tokens. Prioritize key facts, arguments, and conclusions. Maintain coherence and readability.
        
        Target Token Limit: {target_tokens}
        """
        user_prompt = f"Summarize the following content:\n\n{content}"

        try:
            llm_response = await self.llm_client.complete_async(
                agent_id=agent_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=target_tokens,
                temperature=0.3 # Low temperature for factual summarization
            )
            return llm_response.content
        except Exception as e:
            print(f"Error during summarization: {e}. Returning truncated content.")
            # Fallback to simple truncation if LLM summarization fails
            return self._simple_truncate(content, target_tokens)

    def _simple_truncate(self, content: str, target_tokens: int) -> str:
        """
        Performs a simple character-based truncation as a fallback.
        Roughly 1 token = 4 characters for English text.
        """
        char_limit = target_tokens * 4
        if len(content) <= char_limit:
            return content
        return content[:char_limit] + "..."
