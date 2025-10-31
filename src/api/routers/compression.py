"""
Context Compression API router.

Endpoints for compressing large contexts using various strategies.
"""

import logging
import time
import hashlib
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import verify_api_key, get_context_compressor
from src.api.models import (
    # Request models
    CompressionRequest,
    CompressionConfigRequest,
    # Response models
    CompressionResponse,
    CompressionStatsResponse,
    # Enums
    CompressionStrategy,
    CompressionLevel,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/memory/compression",
    tags=["Context Compression"],
    responses={
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_context_size(context: dict) -> int:
    """Calculate size of context in characters."""
    import json
    return len(json.dumps(context))


def generate_context_id(context: dict) -> str:
    """Generate unique ID for compressed context."""
    import json
    content_hash = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    return f"ctx_{content_hash[:12]}"


# ============================================================================
# Endpoint Implementations
# ============================================================================

@router.post("/compress", response_model=CompressionResponse)
async def compress_context(
    request: CompressionRequest,
    api_key: str = Depends(verify_api_key),
    compressor = Depends(get_context_compressor)
):
    """
    Compress context using specified strategy and level.

    Available strategies:
    - **extractive_summary**: Keep important sentences (fast, preserves exact text)
    - **abstractive_summary**: Create brief summaries (concise, loses detail)
    - **keyword_extraction**: Extract key concepts and keywords
    - **hierarchical_summary**: 3-level structure (core/supporting/auxiliary)
    - **relevance_filtering**: Keep only topic-relevant content (requires topic_keywords)
    - **progressive_refinement**: Multiple passes (best quality, slower)

    Compression levels:
    - **light**: 80% of original size
    - **moderate**: 60% of original size
    - **heavy**: 40% of original size
    - **extreme**: 20% of original size
    """
    try:
        from src.memory.context_compression import CompressionStrategy as MemCompressionStrategy
        from src.memory.context_compression import CompressionLevel as MemCompressionLevel

        start_time = time.time()

        # Calculate original size
        original_size = calculate_context_size(request.context)

        # Map API enums to memory facade enums
        strategy_mapping = {
            CompressionStrategy.EXTRACTIVE_SUMMARY: MemCompressionStrategy.EXTRACTIVE_SUMMARY,
            CompressionStrategy.ABSTRACTIVE_SUMMARY: MemCompressionStrategy.ABSTRACTIVE_SUMMARY,
            CompressionStrategy.KEYWORD_EXTRACTION: MemCompressionStrategy.KEYWORD_EXTRACTION,
            CompressionStrategy.HIERARCHICAL_SUMMARY: MemCompressionStrategy.HIERARCHICAL_SUMMARY,
            CompressionStrategy.RELEVANCE_FILTERING: MemCompressionStrategy.RELEVANCE_FILTERING,
            CompressionStrategy.PROGRESSIVE_REFINEMENT: MemCompressionStrategy.PROGRESSIVE_REFINEMENT,
        }

        level_mapping = {
            CompressionLevel.LIGHT: MemCompressionLevel.LIGHT,
            CompressionLevel.MODERATE: MemCompressionLevel.MODERATE,
            CompressionLevel.HEAVY: MemCompressionLevel.HEAVY,
            CompressionLevel.EXTREME: MemCompressionLevel.EXTREME,
        }

        mem_strategy = strategy_mapping[request.strategy]
        mem_level = level_mapping[request.level]

        # Map compression level to target size
        level_to_ratio = {
            MemCompressionLevel.LIGHT: 0.8,
            MemCompressionLevel.MODERATE: 0.6,
            MemCompressionLevel.HEAVY: 0.4,
            MemCompressionLevel.EXTREME: 0.2,
        }
        target_ratio = level_to_ratio.get(mem_level, 0.6)
        target_size = int(original_size * target_ratio)

        # Compress context
        compressed_context = compressor.compress_context(
            context=request.context,
            target_size=target_size,
            strategy=mem_strategy
        )

        # Calculate compressed size
        compressed_size = len(str(compressed_context.content))
        compression_ratio = compressed_size / original_size if original_size > 0 else 0.0

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Generate context ID
        context_id = generate_context_id(request.context)

        return CompressionResponse(
            context_id=context_id,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            strategy_used=request.strategy,
            level_used=request.level,
            compressed_content=compressed_context.content,
            relevance_scores=compressed_context.relevance_scores,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.exception("Error compressing context")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error compressing context: {str(e)}"
        )


@router.get("/stats", response_model=CompressionStatsResponse)
async def get_compression_stats(
    api_key: str = Depends(verify_api_key),
    compressor = Depends(get_context_compressor)
):
    """
    Get compression system statistics and configuration.

    Returns information about available strategies, levels, and current settings.
    """
    try:
        stats = compressor.get_compression_stats()

        from src.memory.context_compression import CompressionStrategy as MemCompressionStrategy
        from src.memory.context_compression import CompressionLevel as MemCompressionLevel

        # Map memory facade enums to API enums
        strategy_mapping = {
            MemCompressionStrategy.EXTRACTIVE_SUMMARY: CompressionStrategy.EXTRACTIVE_SUMMARY,
            MemCompressionStrategy.ABSTRACTIVE_SUMMARY: CompressionStrategy.ABSTRACTIVE_SUMMARY,
            MemCompressionStrategy.KEYWORD_EXTRACTION: CompressionStrategy.KEYWORD_EXTRACTION,
            MemCompressionStrategy.HIERARCHICAL_SUMMARY: CompressionStrategy.HIERARCHICAL_SUMMARY,
            MemCompressionStrategy.RELEVANCE_FILTERING: CompressionStrategy.RELEVANCE_FILTERING,
            MemCompressionStrategy.PROGRESSIVE_REFINEMENT: CompressionStrategy.PROGRESSIVE_REFINEMENT,
        }

        level_mapping = {
            MemCompressionLevel.LIGHT: CompressionLevel.LIGHT,
            MemCompressionLevel.MODERATE: CompressionLevel.MODERATE,
            MemCompressionLevel.HEAVY: CompressionLevel.HEAVY,
            MemCompressionLevel.EXTREME: CompressionLevel.EXTREME,
        }

        # Convert default strategy/level from strings to enums
        try:
            mem_default_strategy = MemCompressionStrategy(stats.get("default_strategy", "hierarchical_summary"))
            default_strategy = strategy_mapping.get(mem_default_strategy, CompressionStrategy.HIERARCHICAL_SUMMARY)
        except (ValueError, KeyError):
            default_strategy = CompressionStrategy.HIERARCHICAL_SUMMARY

        try:
            mem_default_level = MemCompressionLevel(stats.get("default_level", "moderate"))
            default_level = level_mapping.get(mem_default_level, CompressionLevel.MODERATE)
        except (ValueError, KeyError):
            default_level = CompressionLevel.MODERATE

        return CompressionStatsResponse(
            max_context_size=stats.get("max_context_size", 4000),
            default_strategy=default_strategy,
            default_level=default_level,
            available_strategies=[
                CompressionStrategy.EXTRACTIVE_SUMMARY,
                CompressionStrategy.ABSTRACTIVE_SUMMARY,
                CompressionStrategy.KEYWORD_EXTRACTION,
                CompressionStrategy.HIERARCHICAL_SUMMARY,
                CompressionStrategy.RELEVANCE_FILTERING,
                CompressionStrategy.PROGRESSIVE_REFINEMENT,
            ],
            available_levels=[
                CompressionLevel.LIGHT,
                CompressionLevel.MODERATE,
                CompressionLevel.HEAVY,
                CompressionLevel.EXTREME,
            ]
        )

    except Exception as e:
        logger.exception("Error getting compression stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting compression stats: {str(e)}"
        )


@router.patch("/config")
async def update_compression_config(
    request: CompressionConfigRequest,
    api_key: str = Depends(verify_api_key),
    compressor = Depends(get_context_compressor)
):
    """
    Update compression system configuration.

    Allows changing default strategy, level, and max context size.
    """
    try:
        from src.memory.context_compression import CompressionStrategy as MemCompressionStrategy
        from src.memory.context_compression import CompressionLevel as MemCompressionLevel

        # Map API enums to memory facade enums
        strategy_mapping = {
            CompressionStrategy.EXTRACTIVE_SUMMARY: MemCompressionStrategy.EXTRACTIVE_SUMMARY,
            CompressionStrategy.ABSTRACTIVE_SUMMARY: MemCompressionStrategy.ABSTRACTIVE_SUMMARY,
            CompressionStrategy.KEYWORD_EXTRACTION: MemCompressionStrategy.KEYWORD_EXTRACTION,
            CompressionStrategy.HIERARCHICAL_SUMMARY: MemCompressionStrategy.HIERARCHICAL_SUMMARY,
            CompressionStrategy.RELEVANCE_FILTERING: MemCompressionStrategy.RELEVANCE_FILTERING,
            CompressionStrategy.PROGRESSIVE_REFINEMENT: MemCompressionStrategy.PROGRESSIVE_REFINEMENT,
        }

        level_mapping = {
            CompressionLevel.LIGHT: MemCompressionLevel.LIGHT,
            CompressionLevel.MODERATE: MemCompressionLevel.MODERATE,
            CompressionLevel.HEAVY: MemCompressionLevel.HEAVY,
            CompressionLevel.EXTREME: MemCompressionLevel.EXTREME,
        }

        # Build update kwargs
        update_kwargs = {}

        if request.max_context_size is not None:
            update_kwargs['max_context_size'] = request.max_context_size

        if request.strategy is not None:
            update_kwargs['strategy'] = strategy_mapping[request.strategy]

        if request.level is not None:
            update_kwargs['level'] = level_mapping[request.level]

        if request.preserve_keywords is not None:
            update_kwargs['preserve_keywords'] = request.preserve_keywords

        # Update config
        compressor.update_config(**update_kwargs)

        return {
            "message": "Compression configuration updated successfully",
            "config": {
                "max_context_size": compressor.config.max_context_size,
                "strategy": compressor.config.strategy.value,
                "level": compressor.config.level.value,
                "preserve_keywords": compressor.config.preserve_keywords
            }
        }

    except Exception as e:
        logger.exception("Error updating compression config")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating compression config: {str(e)}"
        )
