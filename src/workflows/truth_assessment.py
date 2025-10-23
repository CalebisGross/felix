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

    Args:
        query: User query string

    Returns:
        QueryType enum value
    """
    query_lower = query.lower()

    # Time-sensitive queries
    time_patterns = [
        r'\bcurrent (time|date)\b',
        r'\bwhat time is it\b',
        r'\bwhat is the time\b',
        r'\btime now\b',
    ]
    if any(re.search(pattern, query_lower) for pattern in time_patterns):
        return QueryType.TIME

    # Date queries
    date_patterns = [
        r'\bcurrent date\b',
        r'\btoday.*date\b',
        r'\bwhat.*date\b',
    ]
    if any(re.search(pattern, query_lower) for pattern in date_patterns):
        return QueryType.DATE

    # Current events
    event_patterns = [
        r'\blatest\b',
        r'\brecent\b',
        r'\bcurrent.*news\b',
        r'\bwho won\b',
    ]
    if any(re.search(pattern, query_lower) for pattern in event_patterns):
        return QueryType.CURRENT_EVENT

    # Analysis queries
    analysis_patterns = [
        r'\banalyze\b',
        r'\bevaluate\b',
        r'\bcompare\b',
        r'\bwhy\b',
        r'\bhow does\b',
    ]
    if any(re.search(pattern, query_lower) for pattern in analysis_patterns):
        return QueryType.ANALYSIS

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

    # Generic phrases indicate non-specific information
    generic_phrases = [
        'tools like',
        'websites such as',
        'you can check',
        'provides information',
        'displays the',
        'shows the',
        'no specific',
        'not provided',
        'varies by',
        'depends on',
    ]

    if any(phrase in text_lower for phrase in generic_phrases):
        return False

    # Specific indicators: actual dates, times, numbers
    specific_patterns = [
        r'\b20\d{2}\b',  # Year (2020-2099)
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Month names
        r'\b\d{1,2}:\d{2}\b',  # Time (HH:MM)
        r'\b(am|pm|edt|est|utc|gmt)\b',  # Time zone/period
        r'\b\d+\.\d+\b',  # Decimal numbers
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Date format
    ]

    matches = sum(1 for pattern in specific_patterns if re.search(pattern, text_lower))

    # Need at least 2 specific indicators for time/date queries
    return matches >= 2


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

    # Extract dates from entries
    dates = []
    times = []

    for entry in knowledge_entries:
        if not hasattr(entry, 'content'):
            continue

        content = entry.content
        if isinstance(content, dict):
            text = content.get('result', str(content))
        else:
            text = str(content)

        # Extract years
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        if year_matches:
            dates.extend(year_matches)

        # Extract months
        month_matches = re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text.lower())
        if month_matches:
            dates.extend(month_matches)

        # Extract day numbers
        day_matches = re.findall(r'\b([1-9]|[12][0-9]|3[01])\b', text)
        if day_matches:
            dates.extend(day_matches)

    # Check for significant contradictions
    # Different years in same query is a contradiction
    unique_years = set(y for y in dates if len(y) == 4)
    if len(unique_years) > 1:
        # Allow for year boundaries (Dec 31 vs Jan 1)
        years_list = [int(y) for y in unique_years]
        if max(years_list) - min(years_list) > 1:
            return (True, f"Contradictory years: {', '.join(unique_years)}")

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

    # Extract key facts from each entry
    facts = []
    for entry in knowledge_entries:
        if not hasattr(entry, 'content'):
            continue

        content = entry.content
        if isinstance(content, dict):
            text = content.get('result', str(content))
        else:
            text = str(content)

        # Extract structured facts
        entry_facts = {
            'years': set(re.findall(r'\b(20\d{2})\b', text)),
            'months': set(re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text.lower())),
            'days': set(re.findall(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', text.lower())),
        }
        facts.append(entry_facts)

    # Check agreement on years
    all_years = [f['years'] for f in facts if f['years']]
    if len(all_years) >= 2:
        # All sources should agree on year
        first_years = all_years[0]
        if not all(years == first_years or not years for years in all_years):
            return False

    # Check agreement on months
    all_months = [f['months'] for f in facts if f['months']]
    if len(all_months) >= 2:
        first_months = all_months[0]
        if not all(months == first_months or not months for months in all_months):
            return False

    # Check agreement on day of week
    all_days = [f['days'] for f in facts if f['days']]
    if len(all_days) >= 2:
        first_days = all_days[0]
        if not all(days == first_days or not days for days in all_days):
            return False

    return True


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
            age = current_time - entry.created_at
            if age <= max_age:
                fresh_entries.append(entry)

    if not fresh_entries:
        oldest_age = min((current_time - e.created_at) / 60 for e in high_conf_entries if hasattr(e, 'created_at'))
        return (False, 0.0, f"Knowledge too old ({oldest_age:.1f} minutes, limit {max_age/60:.1f} minutes)")

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

        if deep_search_used and is_authoritative:
            return (True, 0.85, "Single authoritative source with deep search verification")
        elif deep_search_used:
            return (True, 0.75, "Single source with deep search verification")
        elif is_authoritative:
            return (False, 0.0, "Single authoritative source but only snippet data")
        else:
            return (False, 0.0, "Single source without deep verification")

    return (False, 0.0, "Unknown assessment path")
