#!/usr/bin/env python3
"""
Quick troubleshooting script for Knowledge Brain API
"""

import sys
import os

try:
    import httpx
except ImportError:
    print("Error: httpx not installed")
    print("Run: pip install httpx")
    sys.exit(1)

API_URL = "http://localhost:8000"
API_KEY = None  # No authentication for testing

def test_connection():
    """Test basic API connection"""
    print("\n=== Testing API Connection ===")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        print(f"✅ API is reachable: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        print("   Make sure API server is running:")
        print("   python -m uvicorn src.api.main:app --reload --port 8000")
        return False

def test_knowledge_brain_status():
    """Test if Knowledge Brain is enabled"""
    print("\n=== Testing Knowledge Brain Status ===")

    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        # Try to list documents (simplest KB endpoint)
        response = httpx.get(
            f"{API_URL}/api/v1/knowledge/documents",
            headers=headers,
            timeout=5.0
        )

        if response.status_code == 200:
            print("✅ Knowledge Brain is enabled and working")
            result = response.json()
            print(f"   Found {result['total']} documents")
            return True
        elif response.status_code == 503:
            error = response.json()
            print(f"❌ Knowledge Brain is NOT enabled")
            print(f"   Error: {error.get('detail', 'Unknown error')}")
            print("\n   Solution:")
            print("   1. Stop the API server (Ctrl+C)")
            print("   2. Set environment variable:")
            print("      export FELIX_ENABLE_KNOWLEDGE_BRAIN=true")
            print("   3. Restart API server:")
            print("      python -m uvicorn src.api.main:app --reload --port 8000")
            return False
        elif response.status_code == 401:
            print(f"❌ Authentication required")
            print("   Set API_KEY in this script or disable auth:")
            print("   unset FELIX_API_KEY")
            return False
        else:
            print(f"❌ Unexpected error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_file_paths():
    """Check common file path issues"""
    print("\n=== Testing File Paths ===")

    # Check if knowledge_sources directory exists
    knowledge_dir = "/home/hubcaps/Projects/felix/knowledge_sources"
    if os.path.exists(knowledge_dir):
        print(f"✅ Knowledge sources directory exists: {knowledge_dir}")
        files = os.listdir(knowledge_dir)
        if files:
            print(f"   Contains {len(files)} files")
        else:
            print(f"   Directory is empty - add some PDF/TXT files for testing")
    else:
        print(f"ℹ️  Knowledge sources directory doesn't exist: {knowledge_dir}")
        print(f"   Creating it...")
        try:
            os.makedirs(knowledge_dir, exist_ok=True)
            print(f"   ✅ Created: {knowledge_dir}")
            print(f"   Add some PDF or TXT files here for testing")
        except Exception as e:
            print(f"   ❌ Failed to create: {e}")

def test_dependencies():
    """Check if required packages are installed"""
    print("\n=== Testing Dependencies ===")

    required = {
        'httpx': 'httpx',
        'fastapi': 'fastapi',
        'pydantic': 'pydantic',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing.append(package)

    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")

def main():
    print("=" * 60)
    print("Felix Knowledge Brain API - Troubleshooting")
    print("=" * 60)

    # Run tests
    api_ok = test_connection()
    if not api_ok:
        print("\n❌ API server is not running. Start it first.")
        return

    kb_ok = test_knowledge_brain_status()
    test_file_paths()
    test_dependencies()

    # Summary
    print("\n" + "=" * 60)
    if kb_ok:
        print("✅ All checks passed! You can now use the Knowledge Brain API")
        print("\nTry these commands:")
        print("  python examples/api_examples/knowledge_brain_client.py")
        print("  open examples/api_examples/knowledge_brain_demo.html")
    else:
        print("❌ Some issues found. Please fix them and try again.")
    print("=" * 60)

if __name__ == "__main__":
    main()
