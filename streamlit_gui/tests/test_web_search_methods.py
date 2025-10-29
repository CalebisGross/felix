"""Test script for web search monitoring methods in DatabaseReader."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit_gui.backend.db_reader import DatabaseReader

def test_web_search_methods():
    """Test the new web search monitoring methods."""
    print("Testing Web Search Monitoring Methods")
    print("=" * 60)

    # Initialize DatabaseReader
    reader = DatabaseReader()

    # Test 1: get_web_search_activity
    print("\n1. Testing get_web_search_activity()")
    print("-" * 60)
    df = reader.get_web_search_activity(limit=10)
    print(f"Returned DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if not df.empty:
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print("\nNo web search activity found in database.")

    # Test 2: get_web_search_stats
    print("\n\n2. Testing get_web_search_stats()")
    print("-" * 60)
    stats = reader.get_web_search_stats()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Tests completed successfully!")

    return df, stats

if __name__ == "__main__":
    try:
        df, stats = test_web_search_methods()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
