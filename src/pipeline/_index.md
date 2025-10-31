# Pipeline Module

## Purpose
Data processing pipelines providing streaming, chunking, and progressive processing for efficient handling of large LLM outputs and real-time token delivery.

## Key Files

### [chunking.py](chunking.py)
Token-level streaming and progressive processing.

**Key Classes**:

#### `ChunkedResult`
Structure for progressive processing results.
- **content**: Accumulated content so far
- **is_complete**: Processing completion flag
- **metadata**: Additional context (token count, chunks processed, etc.)
- **chunks**: List of individual chunks

#### `ProgressiveProcessor`
Processes content incrementally as it becomes available.
- **Purpose**: Enable agent processing before full LLM response completes
- **Benefits**: Reduced latency, early validation, faster feedback

**Key Methods**:
- `add_chunk(chunk: str)`: Add new content chunk
- `get_current_state() -> ChunkedResult`: Get processing state
- `is_ready_for_processing() -> bool`: Check if enough content available
- `process_available()`: Process accumulated chunks

#### `ContentSummarizer`
Summarizes content progressively.
- **Purpose**: Create rolling summaries of streaming content
- **Use cases**: Progress indicators, early synthesis, context compression

**Key Methods**:
- `add_content(text: str)`: Add new content
- `get_summary() -> str`: Get current summary
- `update_summary()`: Refresh summary with new content

## Key Concepts

### Progressive Processing

Traditional approach:
```
LLM generates full response (30s)
       ↓
Agent receives complete text
       ↓
Agent begins processing
       ↓
Total latency: 30s + processing time
```

Progressive approach:
```
LLM generates chunk 1 (2s)
       ↓
Agent begins processing chunk 1
       ↓
LLM generates chunk 2 (2s)
       ↓
Agent processes chunk 2 (parallel)
       ↓
...streaming continues...
       ↓
Total latency: ~2s + processing time
```

### Chunk Size

Default chunk size: 512 tokens
- Small enough: Low latency per chunk
- Large enough: Sufficient context for processing
- Configurable: Adjust based on model and task

### Streaming Architecture

```
LLM Studio
    ↓ (token stream)
LMStudioClient
    ↓ (time-batched chunks)
ProgressiveProcessor
    ↓ (processed chunks)
Agent / Synthesis
```

### Use Cases

#### 1. Early Validation
- Process first chunks for fact-checking
- Detect errors early and abort if needed
- Save tokens and time on invalid responses

#### 2. Progressive Summarization
- Update summary as content streams
- Show progress indicators to user
- Enable early decisions on relevance

#### 3. Parallel Processing
- Different agents process different chunks
- Reduce overall latency through parallelism
- Merge results at end

#### 4. Context Compression
- Compress chunks as they arrive
- Keep working memory bounded
- Maintain essential information only

### Chunking Strategies

#### Fixed-Size Chunking
- Split at regular token intervals (e.g., 512 tokens)
- Simple and predictable
- May split sentences/thoughts

#### Semantic Chunking
- Split at natural boundaries (sentences, paragraphs)
- Preserves meaning
- Variable chunk sizes

#### Hybrid Chunking (default)
- Prefer semantic boundaries
- Fall back to fixed size if needed
- Best of both approaches

## Configuration

```yaml
pipeline:
  chunk_size: 512                     # Tokens per chunk
  chunking_strategy: "hybrid"         # fixed/semantic/hybrid
  enable_progressive: true            # Enable progressive processing
  min_chunk_for_processing: 256       # Min tokens before processing
  summary_update_frequency: 3         # Update summary every N chunks
```

## Usage Example

```python
from src.pipeline.chunking import ProgressiveProcessor, ContentSummarizer

# Initialize progressive processor
processor = ProgressiveProcessor(chunk_size=512)

# Simulating streaming LLM response
for chunk in llm_stream():
    # Add chunk to processor
    processor.add_chunk(chunk)

    # Check if ready for processing
    if processor.is_ready_for_processing():
        # Process available content
        result = processor.process_available()
        print(f"Processed {len(result.chunks)} chunks")

# Final processing
final_result = processor.get_current_state()
if final_result.is_complete:
    print("Processing complete!")
```

### Summarization Example

```python
from src.pipeline.chunking import ContentSummarizer

summarizer = ContentSummarizer()

# Stream content
for chunk in llm_stream():
    summarizer.add_content(chunk)

    # Get rolling summary
    summary = summarizer.get_summary()
    print(f"Current summary: {summary}")

# Final summary
final_summary = summarizer.get_summary()
```

## Integration Points

### LLM Streaming
- [LMStudioClient](../llm/lm_studio_client.py) produces token stream
- Time-batched delivery (0.1s intervals)
- ProgressiveProcessor consumes stream

### Agent Processing
- Agents can start processing before response complete
- Reduces perceived latency
- Enables early feedback

### GUI Display
- Terminal tab shows streaming output
- Progress indicators update in real-time
- User sees immediate feedback

## Performance Benefits

**Latency Reduction**:
- Traditional: Full generation time + processing time
- Progressive: First chunk latency + processing time
- Improvement: Often 10-20x faster perceived response

**Memory Efficiency**:
- Process and discard chunks
- Bounded memory usage
- No need to buffer entire response

**User Experience**:
- Immediate feedback
- Progress visibility
- Can cancel early if wrong direction

## Related Modules
- [llm/](../llm/) - LMStudioClient provides streaming tokens
- [agents/](../agents/) - Agents consume progressive results
- [communication/](../communication/) - StreamingCoordinator uses chunking
- [gui/](../gui/) - Terminal tab displays streaming output
