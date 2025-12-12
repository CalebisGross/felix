"""
Truth Assessment Module for Felix Framework

Evaluates the trustability of knowledge entries to determine if answers
can be trusted without additional validation from Analysis agents.

This module supports the adaptive confidence threshold system by assessing:
- Knowledge quality and consensus
- Source authority and freshness
- Contradiction detection
- Specificity of information
"""

import logging
import time
import re
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger('felix_workflows')


class QueryType(Enum):
    """Types of queries for freshness assessment."""
    TIME = "time"
    DATE = "date"
    CURRENT_EVENT = "current_event"
    GENERAL_FACT = "general_fact"
    ANALYSIS = "analysis"


def detect_query_type(query: str) -> QueryType:
    """
    Detect the type of query to determine freshness requirements.

    For composite queries (containing multiple types), returns the MOST SENSITIVE
    type to ensure all parts of the query have appropriately fresh data.

    Sensitivity hierarchy (most to least strict):
    TIME (5min) > DATE (1hr) > CURRENT_EVENT (30min) > ANALYSIS (24hr) > GENERAL_FACT (24hr)

    Args:
        query: User query string

    Returns:
        QueryType enum value (most sensitive type detected)
    """
    query_lower = query.lower()
    detected_types = []

    # TIME queries - most sensitive (5 minute freshness)
    # More specific patterns to avoid false positives
    time_patterns = [
        r'\bcurrent time\b',
        r'\bwhat time is it\b',
        r'\bwhat is the time\b',
        r'\btime now\b',
        r'\btime right now\b',
        r'\bwhat.*time(?!\s+(period|frame|zone|difference))',  # Avoid "time period", "time frame", etc.
    ]
    if any(re.search(pattern, query_lower) for pattern in time_patterns):
        detected_types.append(QueryType.TIME)

    # DATE queries - strict (1 hour freshness)
    # More specific patterns - must be about "the date" not just contain "date"
    date_patterns = [
        r'\bcurrent date\b',
        r'\btoday\'?s?\s+date\b',
        r'\bwhat\s+(is\s+)?the\s+date\b',  # "what is the date" or "what the date"
        r'\bdate\s+(is\s+it|today|now)\b',  # "date is it", "date today", "date now"
    ]
    if any(re.search(pattern, query_lower) for pattern in date_patterns):
        detected_types.append(QueryType.DATE)

    # CURRENT_EVENT queries - moderate (30 minute freshness)
    event_patterns = [
        r'\blatest\b',
        r'\brecent\b',
        r'\bcurrent.*news\b',
        r'\bwho won\b',
        r'\bjust happened\b',
        r'\bbreaking\b',
    ]
    if any(re.search(pattern, query_lower) for pattern in event_patterns):
        detected_types.append(QueryType.CURRENT_EVENT)

    # ANALYSIS queries - lenient (24 hour freshness)
    analysis_patterns = [
        r'\banalyze\b',
        r'\bevaluate\b',
        r'\bcompare\b',
        r'\bwhy\b',
        r'\bhow does\b',
        r'\bexplain\b',
    ]
    if any(re.search(pattern, query_lower) for pattern in analysis_patterns):
        detected_types.append(QueryType.ANALYSIS)

    # Sensitivity hierarchy: return most sensitive type detected
    # TIME is most sensitive (5 min), so check it first
    if QueryType.TIME in detected_types:
        if len(detected_types) > 1:
            logger.info(f"📊 Composite query detected: {detected_types} → Using TIME (most sensitive)")
        return QueryType.TIME

    if QueryType.DATE in detected_types:
        if len(detected_types) > 1:
            logger.info(f"📊 Composite query detected: {detected_types} → Using DATE")
        return QueryType.DATE

    if QueryType.CURRENT_EVENT in detected_types:
        if len(detected_types) > 1:
            logger.info(f"📊 Composite query detected: {detected_types} → Using CURRENT_EVENT")
        return QueryType.CURRENT_EVENT

    if QueryType.ANALYSIS in detected_types:
        return QueryType.ANALYSIS

    # Default to GENERAL_FACT for queries not matching any specific pattern
    return QueryType.GENERAL_FACT


def is_authoritative_domain(knowledge_entry: Any) -> bool:
    """
    Check if knowledge comes from an authoritative domain.

    Args:
        knowledge_entry: Knowledge entry to check

    Returns:
        True if from authoritative source
    """
    if not hasattr(knowledge_entry, 'content') or not isinstance(knowledge_entry.content, dict):
        return False

    source_url = knowledge_entry.content.get('source_url', '')

    # Authoritative time sources
    time_sources = [
        'time.is',
        'timeanddate.com',
        'worldtimeserver.com',
        'currenttime.org',
    ]

    # Check if URL contains authoritative domain
    return any(domain in source_url.lower() for domain in time_sources)


def is_specific_data(content: Any) -> bool:
    """
    Check if content contains specific factual data vs generic information.

    Specific: "October 23, 2025, 1:24 PM"
    Generic: "Tools like time.is provide current time information"

    Args:
        content: Content to check (dict, str, or other)

    Returns:
        True if contains specific data
    """
    # Extract text from content
    if isinstance(content, dict):
        text = content.get('result', str(content))
    else:
        text = str(content)

    text_lower = text.lower()

    # Specific indicators: actual dates, times, numbers
    specific_patterns = [
        r'\b20\d{2}\b',  # Year (2020-2099)
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Month names
        r'\b\d{1,2}:\d{2}\b',  # Time (HH:MM)
        r'\b(am|pm|edt|est|pst|cst|mst|utc|gmt)\b',  # Time zone/period
        r'\b\d+\.\d+\b',  # Decimal numbers
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Date format
    ]

    matches = sum(1 for pattern in specific_patterns if re.search(pattern, text_lower))

    # FIX: Lower threshold - even 1 strong indicator (like "3:45 PM EDT") is specific
    # Check if we have specific data first
    has_specific_data = matches >= 1

    # Generic phrases that indicate LACK of specific information
    # Only reject if we have generic phrases AND no specific data
    generic_only_phrases = [
        'no specific',
        'not provided',
        'not available',
        'cannot determine',
        'unable to provide',
    ]

    # Check for purely generic responses
    is_generic_only = any(phrase in text_lower for phrase in generic_only_phrases)

    # Return True if we have specific data, even if some generic phrases exist
    # (e.g., "According to time.is, it's 3:45 PM EDT" has both but is still specific)
    return has_specific_data and not is_generic_only


def detect_contradictions(knowledge_entries: List[Any]) -> Tuple[bool, Optional[str]]:
    """
    Detect if knowledge entries contain contradictory information.

    Args:
        knowledge_entries: List of knowledge entries to check

    Returns:
        (has_contradictions: bool, reason: str or None)
    """
    if len(knowledge_entries) < 2:
        return (False, None)

    # Extract temporal data from entries with context awareness
    entry_data = []

    for entry in knowledge_entries:
        if not hasattr(entry, 'content'):
            continue

        content = entry.content
        if isinstance(content, dict):
            text = content.get('result', str(content))
            source_url = content.get('source_url', '')
        else:
            text = str(content)
            source_url = ''

        text_lower = text.lower()

        # Check if this is from a programming/code source
        is_code_source = any(domain in source_url.lower() for domain in [
            'stackoverflow.com',
            'github.com',
            'stackexchange.com',
            'gitlab.com',
        ])

        # Extract years (only 4-digit years), but filter out code examples
        year_matches = re.finditer(r'\b(20\d{2})\b', text)
        years = set()

        for match in year_matches:
            year = match.group(1)
            # Get context around the year (50 chars before and after)
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]

            # Skip if year appears in code-like patterns
            is_in_code_pattern = any([
                'datetime.datetime(' in context,  # Python datetime constructor
                'Date(' in context,               # JavaScript Date constructor
                'new Date(' in context,           # JavaScript new Date
                f'({year},' in context,           # Function call with year as first arg
                f', {year},' in context,          # Year as middle argument
            ])

            # If from code source (like StackOverflow) and in code pattern, skip it
            if is_code_source and is_in_code_pattern:
                logger.debug(f"   Ignoring year {year} from code example in {source_url}")
                continue

            years.add(year)

        # Extract months (with context - must be near other date indicators)
        months = set()
        for match in re.finditer(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text_lower):
            # Check if month appears in date context (near numbers or years)
            context_start = max(0, match.start() - 50)
            context_end = min(len(text_lower), match.end() + 50)
            context = text_lower[context_start:context_end]

            # Only count if near year or day number
            if re.search(r'\b20\d{2}\b', context) or re.search(r'\b\d{1,2}[,\s]', context):
                months.add(match.group(1))

        # FIX: Only extract day numbers in date contexts
        # Look for patterns like "October 23" or "23rd" or "23, 2025"
        days = set()
        for match in re.finditer(r'\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b', text):
            context_start = max(0, match.start() - 30)
            context_end = min(len(text), match.end() + 30)
            context = text[context_start:context_end].lower()

            # Only count if near month name or comma/year
            if any(month in context for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                    'july', 'august', 'september', 'october', 'november', 'december']):
                days.add(match.group(1))

        # Get entry metadata for recency weighting
        created_at = getattr(entry, 'created_at', 0)
        entry_source = getattr(entry, 'domain', 'unknown')

        entry_data.append({
            'years': years,
            'months': months,
            'days': days,
            'text': text[:100],  # For debugging
            'created_at': created_at,
            'source': entry_source,
            'source_url': source_url,
            'is_code_source': is_code_source
        })

    # Check for contradictions with recency weighting and majority voting
    import time as time_module
    current_time = time_module.time()

    # Calculate recency weights (exponential decay: weight = e^(-age/300))
    # Entries within 5 minutes get weight ~1.0, 10 min ~0.6, 15 min ~0.4
    for data in entry_data:
        age_seconds = current_time - data['created_at'] if data['created_at'] > 0 else float('inf')
        recency_weight = 2.71828 ** (-age_seconds / 300)  # e^(-age/5min)
        data['recency_weight'] = recency_weight

    all_years = [d['years'] for d in entry_data if d['years']]
    all_months = [d['months'] for d in entry_data if d['months']]
    all_days = [d['days'] for d in entry_data if d['days']]

    # Years: Check for contradiction with recency weighting
    if len(all_years) >= 2:
        # Count weighted votes for each year
        year_votes = {}
        for i, years in enumerate(all_years):
            weight = entry_data[i]['recency_weight']
            for year in years:
                year_votes[year] = year_votes.get(year, 0) + weight

        unique_years = set()
        for years in all_years:
            unique_years.update(years)

        if len(unique_years) > 1:
            years_list = sorted([int(y) for y in unique_years])

            # Check if there's a clear majority with recent entries
            total_weight = sum(year_votes.values())
            if total_weight > 0:
                max_year = max(year_votes, key=year_votes.get)
                max_weight_ratio = year_votes[max_year] / total_weight

                # If one year has >70% of weighted votes, trust it (majority consensus)
                if max_weight_ratio >= 0.7:
                    logger.info(f"   Year consensus: {max_year} ({max_weight_ratio:.0%} weighted votes)")
                    # No contradiction - majority agrees
                    return (False, None)

            # Only flag contradiction if years differ by more than 1 year
            if years_list[-1] - years_list[0] > 1:
                return (True, f"Contradictory years: {', '.join(str(y) for y in years_list)}")

    # Months: Different months in same context is a contradiction
    if len(all_months) >= 2:
        first_months = all_months[0]
        for other_months in all_months[1:]:
            if other_months and first_months and not other_months.intersection(first_months):
                return (True, f"Contradictory months: {first_months} vs {other_months}")

    # Days: Significant day difference (>2 days) is a contradiction
    if len(all_days) >= 2:
        first_days = all_days[0]
        for other_days in all_days[1:]:
            if other_days and first_days:
                # Convert to ints and check difference
                try:
                    first_day_nums = {int(d) for d in first_days}
                    other_day_nums = {int(d) for d in other_days}

                    # If no overlap and difference > 2, it's contradictory
                    if not first_day_nums.intersection(other_day_nums):
                        min_diff = min(abs(d1 - d2) for d1 in first_day_nums for d2 in other_day_nums)
                        if min_diff > 2:
                            return (True, f"Contradictory days: {first_days} vs {other_days}")
                except ValueError:
                    # Skip if conversion fails
                    pass

    return (False, None)


def sources_agree(knowledge_entries: List[Any]) -> bool:
    """
    Check if multiple sources agree on the same information.

    Args:
        knowledge_entries: List of knowledge entries

    Returns:
        True if sources show consensus
    """
    if len(knowledge_entries) < 2:
        return True  # Single source, no disagreement

    # FIX: Use contradiction detection instead of exact string matching
    # If no contradictions detected, sources agree
    has_contradictions, _ = detect_contradictions(knowledge_entries)

    return not has_contradictions


def assess_answer_confidence(knowledge_entries: List[Any], query: str) -> Tuple[bool, float, str]:
    """
    Assess if available knowledge is trustable enough to skip Analysis agent validation.

    Args:
        knowledge_entries: List of knowledge entries from knowledge store
        query: Original user query

    Returns:
        (can_trust: bool, confidence_score: float, reason: str)
    """
    if not knowledge_entries:
        return (False, 0.0, "No knowledge available")

    # Detect query type for freshness requirements
    query_type = detect_query_type(query)

    # Import here to avoid circular dependency
    try:
        from src.memory.knowledge_store import ConfidenceLevel
    except ImportError:
        logger.warning("Could not import ConfidenceLevel, using string comparison")
        ConfidenceLevel = None

    # Filter to HIGH confidence entries from web_search domain
    high_conf_entries = []
    for entry in knowledge_entries:
        if not hasattr(entry, 'confidence_level') or not hasattr(entry, 'domain'):
            continue

        # Check if HIGH confidence
        if ConfidenceLevel:
            is_high = entry.confidence_level == ConfidenceLevel.HIGH
        else:
            is_high = str(entry.confidence_level).upper() == 'HIGH'

        # Check if web_search domain
        is_web_search = entry.domain == "web_search"

        if is_high and is_web_search:
            high_conf_entries.append(entry)

    if not high_conf_entries:
        return (False, 0.0, "No HIGH confidence web_search knowledge")

    # Check freshness based on query type
    current_time = time.time()

    # Freshness requirements (in seconds)
    freshness_limits = {
        QueryType.TIME: 300,           # 5 minutes for time queries
        QueryType.DATE: 3600,          # 1 hour for date queries
        QueryType.CURRENT_EVENT: 1800, # 30 minutes for current events
        QueryType.GENERAL_FACT: 86400, # 24 hours for general facts
        QueryType.ANALYSIS: 86400,     # 24 hours for analysis
    }

    max_age = freshness_limits.get(query_type, 3600)

    fresh_entries = []
    for entry in high_conf_entries:
        if hasattr(entry, 'created_at'):
            # FIX: Handle both Unix timestamp (float) and datetime objects
            try:
                if isinstance(entry.created_at, (int, float)):
                    age = current_time - entry.created_at
                else:
                    # Assume datetime object
                    age = current_time - entry.created_at.timestamp()

                logger.debug(f"Entry age: {age:.1f}s (limit: {max_age}s)")

                if age <= max_age:
                    fresh_entries.append(entry)
            except (AttributeError, TypeError) as e:
                logger.warning(f"Could not parse created_at for entry: {e}")
                # Include entry anyway if we can't parse timestamp (fail-open)
                fresh_entries.append(entry)

    if not fresh_entries:
        # Calculate oldest age for logging
        try:
            oldest_age = min((current_time - e.created_at) / 60 for e in high_conf_entries if hasattr(e, 'created_at') and isinstance(e.created_at, (int, float)))
            return (False, 0.0, f"Knowledge too old ({oldest_age:.1f} minutes, limit {max_age/60:.1f} minutes)")
        except (ValueError, ZeroDivisionError):
            return (False, 0.0, f"Knowledge timestamp format issue or too old (limit {max_age/60:.1f} minutes)")

    # Check for specificity (not generic statements)
    specific_entries = [e for e in fresh_entries if is_specific_data(e.content)]

    if not specific_entries:
        return (False, 0.0, "Only generic information available, no specific facts")

    # Check for contradictions
    has_contradictions, contradiction_reason = detect_contradictions(specific_entries)
    if has_contradictions:
        return (False, 0.0, f"Contradictory information: {contradiction_reason}")

    # Check for consensus (multiple sources agree)
    if len(specific_entries) >= 2:
        if sources_agree(specific_entries):
            return (True, 0.90, f"Consensus from {len(specific_entries)} HIGH confidence sources")
        else:
            return (False, 0.0, "Multiple sources disagree")

    # Single HIGH confidence source
    if len(specific_entries) == 1:
        source = specific_entries[0]

        # Check if deep search was used
        deep_search_used = False
        if isinstance(source.content, dict):
            deep_search_used = source.content.get('deep_search_used', False)

        # Check if authoritative domain
        is_authoritative = is_authoritative_domain(source)

        # FIX: Trust authoritative sources even with snippet data
        # Time sources like time.is are reliable even in snippets
        if deep_search_used and is_authoritative:
            return (True, 0.90, "Single authoritative source with deep search verification")
        elif is_authoritative:
            # FIX: Changed from (False, 0.0) to (True, 0.75)
            # Authoritative time sources are trustworthy even without deep search
            return (True, 0.75, "Single authoritative source with fresh data")
        elif deep_search_used:
            return (True, 0.70, "Single source with deep search verification")
        else:
            # Only reject if non-authoritative AND no deep search
            return (False, 0.0, "Single non-authoritative source without deep verification")

    return (False, 0.0, "Unknown assessment path")

# =============================================================================
# VALIDATION SCORING SYSTEM (Added for Learning & Quality Control)
# =============================================================================

import json  # Already imported at top, but ensuring it's available


def calculate_validation_score(content: Dict[str, Any],
                               source_agent: str,
                               domain: str,
                               confidence_level: Any,
                               existing_knowledge: List[Any] = None) -> float:
    """
    Calculate validation score for a knowledge entry (0.0 to 1.0).

    Higher scores indicate higher quality/trustability.

    Scoring factors:
    - Content specificity (0.2 weight)
    - Source authority (0.2 weight)
    - Contradiction check (0.3 weight)
    - Confidence level (0.2 weight)
    - Content completeness (0.1 weight)

    Args:
        content: Knowledge content dictionary
        source_agent: Agent that generated this knowledge
        domain: Knowledge domain
        confidence_level: Confidence level (enum or string)
        existing_knowledge: Optional existing entries to check contradictions

    Returns:
        Validation score (0.0 = very poor, 1.0 = excellent)
    """
    score = 0.0

    # 1. Content specificity (0.2 weight)
    content_str = json.dumps(content) if isinstance(content, dict) else str(content)

    if is_specific_data(content_str):
        score += 0.2
    else:
        # Partial credit if content has reasonable length
        if len(content_str) > 100:
            score += 0.1
        elif len(content_str) > 20:
            score += 0.05

    # 2. Source authority (0.2 weight)
    if domain == "web_search":
        # Check if content has authoritative domain markers
        if 'source_url' in content:
            source_url = content['source_url']
            if is_authoritative_domain(source_url):
                score += 0.2
            elif any(domain in source_url for domain in ['.edu', '.gov', '.org']):
                score += 0.15
            elif any(domain in source_url for domain in ['.com', '.net']):
                score += 0.1
        else:
            # No source URL - lower score
            score += 0.05
    elif domain == "workflow_task":
        # Agent-generated insights - moderate trust
        score += 0.15
    else:
        # Unknown domain - neutral
        score += 0.1

    # 3. Contradiction check (0.3 weight)
    if existing_knowledge:
        has_contradictions, _ = detect_contradictions(existing_knowledge + [type('obj', (), {'content': content})()])
        if not has_contradictions:
            score += 0.3
        else:
            # Contradictions found - significant penalty
            score += 0.0
    else:
        # No existing knowledge to check against - neutral
        score += 0.15

    # 4. Confidence level (0.2 weight)
    try:
        from src.memory.knowledge_store import ConfidenceLevel
        if confidence_level == ConfidenceLevel.VERIFIED:
            score += 0.2
        elif confidence_level == ConfidenceLevel.HIGH:
            score += 0.15
        elif confidence_level == ConfidenceLevel.MEDIUM:
            score += 0.1
        else:  # LOW
            score += 0.05
    except:
        # Fallback to string comparison
        conf_str = str(confidence_level).upper()
        if 'VERIFIED' in conf_str:
            score += 0.2
        elif 'HIGH' in conf_str:
            score += 0.15
        elif 'MEDIUM' in conf_str:
            score += 0.1
        else:
            score += 0.05

    # 5. Content completeness (0.1 weight)
    if isinstance(content, dict):
        # Check for key fields
        key_fields = ['source_url', 'title', 'text', 'content', 'summary']
        present_fields = sum(1 for field in key_fields if field in content and content[field])
        score += (present_fields / len(key_fields)) * 0.1
    else:
        # Non-dict content gets partial credit if non-empty
        if content:
            score += 0.05

    # Ensure score is in [0.0, 1.0]
    return max(0.0, min(1.0, score))


def get_validation_flags(content: Dict[str, Any],
                        source_agent: str,
                        domain: str,
                        existing_knowledge: List[Any] = None) -> List[str]:
    """
    Get list of validation issues/flags for a knowledge entry.

    Args:
        content: Knowledge content dictionary
        source_agent: Agent that generated this knowledge
        domain: Knowledge domain
        existing_knowledge: Optional existing entries to check contradictions

    Returns:
        List of issue flags (empty if no issues)
    """
    flags = []

    content_str = json.dumps(content) if isinstance(content, dict) else str(content)

    # Check for non-specific data
    if not is_specific_data(content_str):
        flags.append("non_specific_data")

    # Check for missing source
    if domain == "web_search" and 'source_url' not in content:
        flags.append("no_source_citation")

    # Check for very short content
    if len(content_str) < 20:
        flags.append("insufficient_content")

    # Check for contradictions
    if existing_knowledge:
        has_contradictions, details = detect_contradictions(existing_knowledge + [type('obj', (), {'content': content})()])
        if has_contradictions:
            flags.append("temporal_contradiction")

    # Check for generic/placeholder content
    generic_phrases = [
        'not available', 'no information', 'unknown', 'unclear',
        'cannot be determined', 'insufficient data', 'more research needed'
    ]
    content_lower = content_str.lower()
    if any(phrase in content_lower for phrase in generic_phrases):
        flags.append("generic_placeholder")

    # Check for very long content (potential hallucination or irrelevant data dump)
    if len(content_str) > 10000:
        flags.append("excessive_length")

    return flags


def validate_knowledge_entry(content: Dict[str, Any],
                            source_agent: str,
                            domain: str,
                            confidence_level: Any,
                            existing_knowledge: List[Any] = None) -> Dict[str, Any]:
    """
    Validate a knowledge entry and return validation result.

    This is the main validation interface used by KnowledgeStore.

    Args:
        content: Knowledge content dictionary
        source_agent: Agent that generated this knowledge
        domain: Knowledge domain
        confidence_level: Confidence level
        existing_knowledge: Optional existing entries for contradiction checking

    Returns:
        Dictionary with validation results:
        {
            'validation_score': float (0.0-1.0),
            'validation_flags': List[str],
            'validation_status': str ('trusted', 'flagged', 'review', 'quarantine'),
            'should_store': bool
        }
    """
    score = calculate_validation_score(
        content, source_agent, domain, confidence_level, existing_knowledge
    )

    flags = get_validation_flags(
        content, source_agent, domain, existing_knowledge
    )

    # Determine validation status based on score and flags
    if score >= 0.8 and len(flags) == 0:
        status = 'trusted'
        should_store = True
    elif score >= 0.5 and len(flags) <= 2:
        status = 'flagged'
        should_store = True
    elif score >= 0.3 and len(flags) <= 3:
        status = 'review'
        should_store = True
    else:
        status = 'quarantine'
        should_store = False  # Don't store very low quality entries

    return {
        'validation_score': score,
        'validation_flags': flags,
        'validation_status': status,
        'should_store': should_store
    }
