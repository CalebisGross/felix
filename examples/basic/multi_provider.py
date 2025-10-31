"""
Multi-Provider Example

Demonstrates using multiple LLM providers with automatic fallback.
Shows how to configure primary and fallback providers programmatically.
"""

from src.llm.providers import LMStudioProvider, AnthropicProvider, GeminiProvider
from src.llm.llm_router import LLMRouter
from src.llm.base_provider import LLMRequest
import os


def create_router_with_fallbacks():
    """Create router with primary cloud provider and local fallback."""

    providers = []

    # Try to add Anthropic as primary
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            anthropic = AnthropicProvider(
                api_key=anthropic_key,
                model="claude-3-5-sonnet-20241022"
            )
            providers.append(("Anthropic Claude", anthropic))
            print("✓ Anthropic Claude configured as primary")
        except Exception as e:
            print(f"✗ Anthropic not available: {e}")

    # Try to add Gemini as fallback
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            gemini = GeminiProvider(
                api_key=gemini_key,
                model="gemini-1.5-flash-latest"
            )
            providers.append(("Google Gemini", gemini))
            print("✓ Google Gemini configured as fallback #1")
        except Exception as e:
            print(f"✗ Gemini not available: {e}")

    # Always add LM Studio as final fallback
    try:
        lm_studio = LMStudioProvider(base_url="http://localhost:1234/v1")
        providers.append(("LM Studio (local)", lm_studio))
        print("✓ LM Studio configured as fallback #2")
    except Exception as e:
        print(f"✗ LM Studio not available: {e}")

    if not providers:
        raise RuntimeError("No LLM providers available!")

    # Create router
    primary_name, primary = providers[0]
    fallbacks = [p for _, p in providers[1:]]

    router = LLMRouter(
        primary_provider=primary,
        fallback_providers=fallbacks,
        verbose_logging=True
    )

    print(f"\n📡 Router configured with {len(providers)} provider(s)")
    return router


def test_provider_fallback():
    """Test that fallback works when primary fails."""

    print("Multi-Provider Fallback Test")
    print("=" * 50)

    # Create router
    router = create_router_with_fallbacks()

    # Test all connections
    print("\n🔍 Testing connections...")
    results = router.test_all_connections()
    for provider, status in results.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {provider}: {'Connected' if status else 'Offline'}")

    # Make a simple request
    print("\n💬 Testing completion...")
    request = LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello from Felix!' in exactly 5 words.",
        temperature=0.7,
        max_tokens=50
    )

    try:
        response = router.complete(request)
        print(f"\n✓ Response from {response.provider}:")
        print(f"  {response.content}")
        print(f"  Tokens: {response.tokens_used}")
        print(f"  Time: {response.response_time:.2f}s")
    except Exception as e:
        print(f"\n✗ All providers failed: {e}")

    # Show statistics
    print("\n📊 Router Statistics:")
    stats = router.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Primary successes: {stats['primary_successes']}")
    print(f"  Fallback successes: {stats['fallback_successes']}")
    print(f"  Failures: {stats['total_failures']}")
    print(f"  Overall success rate: {stats['overall_success_rate']:.1%}")


def estimate_costs():
    """Show cost estimates for different providers."""

    print("\n" + "=" * 50)
    print("Cost Comparison (per 10K tokens)")
    print("=" * 50)

    # Anthropic pricing
    print("\n🔷 Anthropic Claude 3.5 Sonnet:")
    anthropic_cost = ((10000 / 2) / 1_000_000) * 3.00  # Prompt
    anthropic_cost += ((10000 / 2) / 1_000_000) * 15.00  # Completion
    print(f"  Estimated: ${anthropic_cost:.4f}")

    # Gemini pricing
    print("\n🔶 Google Gemini 1.5 Flash:")
    gemini_cost = ((10000 / 2) / 1_000_000) * 0.075  # Prompt
    gemini_cost += ((10000 / 2) / 1_000_000) * 0.30  # Completion
    print(f"  Estimated: ${gemini_cost:.6f} (very cheap!)")

    # LM Studio (local)
    print("\n⚫ LM Studio (Local):")
    print(f"  Estimated: $0.0000 (free!)")

    print("\n💡 Tip: Use Gemini Flash for high-volume, cost-sensitive tasks")


def main():
    """Run the multi-provider example."""

    try:
        test_provider_fallback()
        estimate_costs()

        print("\n" + "=" * 50)
        print("✓ Multi-provider test complete!")
        print("\nTo use a specific provider, set environment variables:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export GEMINI_API_KEY='your-key'")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure at least one LLM provider is configured:")
        print("  - LM Studio running on localhost:1234")
        print("  - ANTHROPIC_API_KEY environment variable set")
        print("  - GEMINI_API_KEY environment variable set")


if __name__ == "__main__":
    main()
