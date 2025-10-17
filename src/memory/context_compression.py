"""
Context Compression System for the Felix Framework.

Provides intelligent summarization and compression of large contexts,
relevance-based filtering, and progressive context refinement.
"""

import json
import time
import hashlib
import re
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class CompressionStrategy(Enum):
    """Available compression strategies."""
    EXTRACTIVE_SUMMARY = "extractive_summary"
    ABSTRACTIVE_SUMMARY = "abstractive_summary"
    KEYWORD_EXTRACTION = "keyword_extraction"
    HIERARCHICAL_SUMMARY = "hierarchical_summary"
    RELEVANCE_FILTERING = "relevance_filtering"
    PROGRESSIVE_REFINEMENT = "progressive_refinement"

class CompressionLevel(Enum):
    """Compression intensity levels."""
    LIGHT = "light"      # 80% of original
    MODERATE = "moderate"  # 60% of original
    HEAVY = "heavy"      # 40% of original
    EXTREME = "extreme"  # 20% of original

@dataclass
class CompressedContext:
    """Container for compressed context data."""
    context_id: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    compression_level: CompressionLevel
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    relevance_scores: Dict[str, float]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def get_compression_efficiency(self) -> float:
        """Calculate compression efficiency (higher is better)."""
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.compressed_size / self.original_size)

@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    max_context_size: int = 4000  # Maximum tokens to retain
    strategy: CompressionStrategy = CompressionStrategy.HIERARCHICAL_SUMMARY
    level: CompressionLevel = CompressionLevel.MODERATE
    preserve_keywords: List[str] = field(default_factory=list)
    preserve_structure: bool = True
    maintain_coherence: bool = True
    relevance_threshold: float = 0.3
    
class ContextCompressor:
    """
    Intelligent context compression system.
    
    Reduces context size while preserving important information
    through various compression strategies.
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize context compressor.
        
        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()
        self._stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load common stopwords for text processing."""
        return {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'man', 'she', 'use', 'way', 'where', 'much', 'your', 'from',
            'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very',
            'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
            'such', 'take', 'than', 'them', 'well', 'were', 'will', 'with'
        }
    
    def _generate_context_id(self, content: Dict[str, Any]) -> str:
        """Generate unique ID for compressed context."""
        content_str = json.dumps(content, sort_keys=True)
        hash_input = f"{content_str}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _compress_content(self, content: Dict[str, Any]) -> bytes:
        """Compress large content using pickle."""
        return pickle.dumps(content)
    
    def _decompress_content(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress content from bytes."""
        return pickle.loads(compressed_data)
    
    def compress_context(self, context: Dict[str, Any], 
                        target_size: Optional[int] = None,
                        strategy: Optional[CompressionStrategy] = None) -> CompressedContext:
        """
        Compress context using specified strategy.
        
        Args:
            context: Original context to compress
            target_size: Target size in characters (uses config default if None)
            strategy: Compression strategy (uses config default if None)
            
        Returns:
            Compressed context object
        """
        target_size = target_size or self.config.max_context_size
        strategy = strategy or self.config.strategy
        
        original_size = len(json.dumps(context))
        
        # Apply compression strategy
        if strategy == CompressionStrategy.EXTRACTIVE_SUMMARY:
            compressed_content = self._extractive_summary(context, target_size)
        elif strategy == CompressionStrategy.ABSTRACTIVE_SUMMARY:
            compressed_content = self._abstractive_summary(context, target_size)
        elif strategy == CompressionStrategy.KEYWORD_EXTRACTION:
            compressed_content = self._keyword_extraction(context, target_size)
        elif strategy == CompressionStrategy.HIERARCHICAL_SUMMARY:
            compressed_content = self._hierarchical_summary(context, target_size)
        elif strategy == CompressionStrategy.RELEVANCE_FILTERING:
            compressed_content = self._relevance_filtering(context, target_size)
        elif strategy == CompressionStrategy.PROGRESSIVE_REFINEMENT:
            compressed_content = self._progressive_refinement(context, target_size)
        else:
            compressed_content = self._hierarchical_summary(context, target_size)
        
        compressed_size = len(json.dumps(compressed_content['content']))
        compression_ratio = compressed_size / original_size if original_size > 0 else 0.0
        
        return CompressedContext(
            context_id=self._generate_context_id(context),
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            strategy_used=strategy,
            compression_level=self.config.level,
            content=compressed_content['content'],
            metadata=compressed_content['metadata'],
            relevance_scores=compressed_content['relevance_scores']
        )
    
    def _extractive_summary(self, context: Dict[str, Any], 
                          target_size: int) -> Dict[str, Any]:
        """Extract most important sentences/sections."""
        content = {}
        metadata = {'method': 'extractive_summary'}
        relevance_scores = {}
        
        # Score each text section by importance
        for key, value in context.items():
            if isinstance(value, str):
                sentences = self._split_into_sentences(value)
                scored_sentences = []
                
                for sentence in sentences:
                    score = self._calculate_sentence_importance(sentence, context)
                    scored_sentences.append((sentence, score))
                    relevance_scores[f"{key}_{len(scored_sentences)}"] = score
                
                # Sort by score and select top sentences
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate how many sentences to keep
                target_sentences = max(1, len(scored_sentences) // 3)
                selected_sentences = [s[0] for s in scored_sentences[:target_sentences]]
                
                content[key] = " ".join(selected_sentences)
            else:
                content[key] = value
        
        return {
            'content': content,
            'metadata': metadata,
            'relevance_scores': relevance_scores
        }
    
    def _abstractive_summary(self, context: Dict[str, Any], 
                           target_size: int) -> Dict[str, Any]:
        """Create abstractive summaries of content."""
        content = {}
        metadata = {'method': 'abstractive_summary'}
        relevance_scores = {}
        
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 200:
                # Simple abstractive summary (could be enhanced with LLM)
                summary = self._create_abstract_summary(value)
                content[key] = summary
                relevance_scores[key] = 0.8  # High relevance for summaries
            else:
                content[key] = value
                relevance_scores[key] = 0.6
        
        return {
            'content': content,
            'metadata': metadata,
            'relevance_scores': relevance_scores
        }
    
    def _keyword_extraction(self, context: Dict[str, Any], 
                          target_size: int) -> Dict[str, Any]:
        """Extract key terms and concepts."""
        content = {}
        metadata = {'method': 'keyword_extraction'}
        relevance_scores = {}
        
        all_text = " ".join([str(v) for v in context.values() if isinstance(v, str)])
        keywords = self._extract_keywords(all_text)
        
        # Create keyword-focused content
        content['keywords'] = keywords[:20]  # Top 20 keywords
        content['key_concepts'] = self._extract_key_concepts(all_text)
        
        # Preserve structure with shortened content
        for key, value in context.items():
            if isinstance(value, str):
                # Keep sentences with high keyword density
                sentences = self._split_into_sentences(value)
                keyword_sentences = []
                
                for sentence in sentences:
                    keyword_count = sum(1 for kw in keywords[:10] 
                                      if kw.lower() in sentence.lower())
                    if keyword_count > 0:
                        keyword_sentences.append(sentence)
                        relevance_scores[f"{key}_sentence"] = keyword_count / len(keywords[:10])
                
                if keyword_sentences:
                    content[f"{key}_summary"] = " ".join(keyword_sentences[:3])
            else:
                content[key] = value
        
        return {
            'content': content,
            'metadata': metadata,
            'relevance_scores': relevance_scores
        }
    
    def _hierarchical_summary(self, context: Dict[str, Any], 
                            target_size: int) -> Dict[str, Any]:
        """Create hierarchical summary with multiple levels of detail."""
        content = {}
        metadata = {'method': 'hierarchical_summary', 'levels': 3}
        relevance_scores = {}
        
        # Level 1: Core information (highest priority)
        core_info = {}
        for key, value in context.items():
            if key in ['task', 'objective', 'requirements', 'constraints']:
                core_info[key] = value
                relevance_scores[f"core_{key}"] = 1.0
        
        content['core'] = core_info
        
        # Level 2: Supporting details
        supporting_info = {}
        for key, value in context.items():
            if key not in core_info and isinstance(value, str):
                if len(value) > 100:
                    # Summarize long text
                    summary = self._create_brief_summary(value)
                    supporting_info[key] = summary
                    relevance_scores[f"support_{key}"] = 0.7
                else:
                    supporting_info[key] = value
                    relevance_scores[f"support_{key}"] = 0.8
        
        content['supporting'] = supporting_info
        
        # Level 3: Metadata and auxiliary info
        auxiliary_info = {}
        for key, value in context.items():
            if key not in core_info and key not in supporting_info:
                auxiliary_info[key] = value
                relevance_scores[f"aux_{key}"] = 0.5
        
        content['auxiliary'] = auxiliary_info
        
        return {
            'content': content,
            'metadata': metadata,
            'relevance_scores': relevance_scores
        }
    
    def _relevance_filtering(self, context: Dict[str, Any], 
                           target_size: int) -> Dict[str, Any]:
        """Filter content by relevance to main objectives."""
        content = {}
        metadata = {'method': 'relevance_filtering'}
        relevance_scores = {}
        
        # Identify main topics/objectives
        main_topics = self._identify_main_topics(context)
        
        for key, value in context.items():
            if isinstance(value, str):
                relevance = self._calculate_relevance_to_topics(value, main_topics)
                relevance_scores[key] = relevance
                
                if relevance >= self.config.relevance_threshold:
                    content[key] = value
                elif relevance >= self.config.relevance_threshold * 0.5:
                    # Include abbreviated version for moderately relevant content
                    content[f"{key}_brief"] = self._create_brief_summary(value)
            else:
                content[key] = value
                relevance_scores[key] = 0.8  # Assume structured data is relevant
        
        return {
            'content': content,
            'metadata': metadata,
            'relevance_scores': relevance_scores
        }
    
    def _progressive_refinement(self, context: Dict[str, Any], 
                              target_size: int) -> Dict[str, Any]:
        """Apply multiple compression passes for optimal size."""
        content = {}
        metadata = {'method': 'progressive_refinement', 'passes': 0}
        relevance_scores = {}
        
        current_context = context.copy()
        passes = 0
        max_passes = 3
        
        while len(json.dumps(current_context)) > target_size and passes < max_passes:
            passes += 1
            
            if passes == 1:
                # First pass: Remove low-relevance content
                result = self._relevance_filtering(current_context, target_size)
            elif passes == 2:
                # Second pass: Summarize remaining content
                result = self._hierarchical_summary(current_context, target_size)
            else:
                # Final pass: Keyword extraction
                result = self._keyword_extraction(current_context, target_size)
            
            current_context = result['content']
            relevance_scores.update(result['relevance_scores'])
        
        metadata['passes'] = passes
        
        return {
            'content': current_context,
            'metadata': metadata,
            'relevance_scores': relevance_scores
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_sentence_importance(self, sentence: str, 
                                     context: Dict[str, Any]) -> float:
        """Calculate importance score for a sentence."""
        score = 0.0
        words = sentence.lower().split()
        
        # Length bonus (medium-length sentences preferred)
        if 10 <= len(words) <= 25:
            score += 0.2
        
        # Keyword presence bonus
        keywords = self.config.preserve_keywords
        keyword_count = sum(1 for word in words if word in keywords)
        score += keyword_count * 0.3
        
        # Position bonus (first and last sentences often important)
        # This would need context about sentence position
        
        # Complexity bonus (sentences with numbers, technical terms)
        if any(char.isdigit() for char in sentence):
            score += 0.1
        
        if any(word.isupper() for word in words):
            score += 0.1
        
        return score
    
    def _create_abstract_summary(self, text: str) -> str:
        """Create an abstract summary of text."""
        # Simple implementation - could be enhanced with LLM
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 2:
            return text
        
        # Take first sentence and most informative middle sentence
        first = sentences[0]
        
        if len(sentences) > 2:
            middle_sentences = sentences[1:-1]
            if middle_sentences:
                # Choose sentence with most keywords
                best_middle = max(middle_sentences, 
                                key=lambda s: self._calculate_sentence_importance(s, {}))
                return f"{first} {best_middle}"
        
        return first
    
    def _create_brief_summary(self, text: str) -> str:
        """Create a brief summary of text."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return text
        
        # Take first sentence and add key information from others
        summary = sentences[0]
        
        # Extract key numbers, names, and technical terms from other sentences
        key_info = []
        for sentence in sentences[1:]:
            # Extract numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
            key_info.extend(numbers)
            
            # Extract capitalized words (names, acronyms)
            caps = re.findall(r'\b[A-Z][A-Za-z]*\b', sentence)
            key_info.extend(caps[:2])  # Limit to avoid clutter
        
        if key_info:
            summary += f" Key details: {', '.join(set(key_info)[:5])}"
        
        return summary
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        # Filter out stopwords
        keywords = [w for w in words if w not in self._stopwords]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(word_freq.keys(), 
                               key=lambda x: word_freq[x], 
                               reverse=True)
        
        return sorted_keywords[:30]  # Top 30 keywords
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts and technical terms."""
        concepts = []
        
        # Technical patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+[A-Z]\w*\b',  # CamelCase
            r'\b\w+_\w+\b',  # snake_case
            r'\b\d+\.?\d*[a-zA-Z]+\b',  # Numbers with units
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        # Convert to list and remove duplicates, then slice
        unique_concepts = list(set(concepts))
        return unique_concepts[:15]
    
    def _identify_main_topics(self, context: Dict[str, Any]) -> List[str]:
        """Identify main topics from context."""
        topics = []
        
        # Priority keys that usually contain main topics
        priority_keys = ['task', 'objective', 'goal', 'purpose', 'requirements']
        
        for key in priority_keys:
            if key in context and isinstance(context[key], str):
                keywords = self._extract_keywords(context[key])
                topics.extend(keywords[:5])
        
        # If no priority keys found, extract from all text content
        if not topics:
            all_text = " ".join([str(v) for v in context.values() 
                               if isinstance(v, str)])
            topics = self._extract_keywords(all_text)[:10]
        
        return topics
    
    def _calculate_relevance_to_topics(self, text: str, topics: List[str]) -> float:
        """Calculate how relevant text is to main topics."""
        if not topics:
            return 0.5  # Neutral relevance if no topics
        
        text_lower = text.lower()
        matches = sum(1 for topic in topics if topic.lower() in text_lower)
        
        relevance = matches / len(topics)
        
        # Boost relevance for text with multiple topic mentions
        total_mentions = sum(text_lower.count(topic.lower()) for topic in topics)
        if total_mentions > matches:
            relevance += 0.1 * (total_mentions - matches)
        
        return min(1.0, relevance)
    
    def decompress_context(self, compressed_context: CompressedContext) -> Dict[str, Any]:
        """
        Decompress context (limited reconstruction possible).
        
        Args:
            compressed_context: Compressed context object
            
        Returns:
            Decompressed context (may not be identical to original)
        """
        # Update access count
        compressed_context.access_count += 1
        
        # Return the compressed content with metadata about compression
        result = compressed_context.content.copy()
        result['_compression_metadata'] = {
            'original_size': compressed_context.original_size,
            'compressed_size': compressed_context.compressed_size,
            'compression_ratio': compressed_context.compression_ratio,
            'strategy_used': compressed_context.strategy_used.value,
            'compression_level': compressed_context.compression_level.value,
            'relevance_scores': compressed_context.relevance_scores
        }
        
        return result
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get statistics about compression performance."""
        # This would typically track multiple compressions
        # For now, return configuration info
        return {
            'max_context_size': self.config.max_context_size,
            'default_strategy': self.config.strategy.value,
            'default_level': self.config.level.value,
            'preserve_keywords': len(self.config.preserve_keywords),
            'preserve_structure': self.config.preserve_structure,
            'maintain_coherence': self.config.maintain_coherence,
            'relevance_threshold': self.config.relevance_threshold
        }
    
    def update_config(self, **kwargs) -> None:
        """Update compression configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
