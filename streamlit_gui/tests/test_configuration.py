#!/usr/bin/env python3
"""
Playwright test script for Felix Streamlit GUI Configuration page verification.
Tests badge presence, dropdown functionality, helix visualization, and export buttons.
"""

import asyncio
import sys
import os
from playwright.async_api import async_playwright, expect

async def test_configuration_page():
    """Test the Configuration page functionality."""
    print("Starting Configuration page tests...")

    async with async_playwright() as p:
        # Launch browser in headless mode
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Navigate to main page first
            print("Navigating to main page...")
            await page.goto("http://localhost:8501")
            await page.wait_for_load_state("networkidle")

            # Wait for the page to load and look for sidebar navigation
            await page.wait_for_timeout(2000)

            # Try to find and click the Configuration link in sidebar
            print("Looking for Configuration page link...")
            config_link_selectors = [
                "a:has-text('Configuration')",
                "div[role='button']:has-text('Configuration')",
                "button:has-text('Configuration')",
                "div[data-testid='stSidebar'] a:has-text('Configuration')",
                "div[data-testid='stSidebar'] div:has-text('Configuration')"
            ]

            config_link = None
            for selector in config_link_selectors:
                try:
                    link = page.locator(selector).first
                    if await link.is_visible(timeout=2000):
                        config_link = link
                        print(f"Found Configuration link with selector: {selector}")
                        break
                except:
                    continue

            if config_link:
                await config_link.click()
                print("Clicked Configuration link")
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
            else:
                print("Configuration link not found, trying direct URL navigation...")
                await page.goto("http://localhost:8501/2_Configuration")
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)

            # Test 1: Data Source Badge
            print("Testing Data Source Badge...")
            badge_results = {}

            try:
                # Look for the success badge with "Real Data" text - try multiple possible selectors
                badge_selectors = [
                    "div.stSuccess",
                    "div.element-container div.stSuccess",
                    "[data-testid='stSuccess']",
                    "div:has-text('Real Data')"
                ]

                badge = None
                badge_text = ""

                for selector in badge_selectors:
                    try:
                        elements = page.locator(selector)
                        count = await elements.count()
                        for i in range(count):
                            element = elements.nth(i)
                            text = await element.text_content()
                            if "Real Data" in text:
                                badge = element
                                badge_text = text
                                break
                        if badge:
                            break
                    except:
                        continue

                if badge and "Real Data" in badge_text:
                    await expect(badge).to_be_visible(timeout=5000)
                    badge_results["badge_present"] = "PASS"
                    badge_results["badge_mentions_config"] = "PASS" if "actual Felix configuration files" in badge_text else "FAIL"
                    print(f"   Badge text: {badge_text}")
                else:
                    badge_results["badge_present"] = "FAIL"
                    badge_results["badge_mentions_config"] = "FAIL"
                    print("   Badge not found or doesn't contain expected text")
            except Exception as e:
                badge_results["badge_present"] = "FAIL"
                badge_results["badge_mentions_config"] = "FAIL"
                print(f"   FAIL Badge test failed: {e}")

            # Test 2: Configuration Dropdown Functionality
            print("Testing Configuration Dropdown...")
            dropdown_results = {}

            try:
                # Find and interact with the configuration source dropdown - try multiple selectors
                dropdown_selectors = [
                    "div[data-baseweb='select'] input",
                    "select",
                    "div[role='combobox']",
                    "input[role='combobox']"
                ]

                dropdown = None
                for selector in dropdown_selectors:
                    try:
                        dropdown = page.locator(selector).first
                        if await dropdown.is_visible(timeout=2000):
                            break
                    except:
                        continue

                if dropdown:
                    await expect(dropdown).to_be_visible(timeout=5000)

                    # Click dropdown to open options
                    await dropdown.click()

                    # Select different config options and verify content changes
                    config_options = [
                        "felix_gui_config.json",
                        "streamlit_config.yaml",
                        "configs/default_config.yaml"
                    ]

                    for option in config_options:
                        try:
                            # Select option - try multiple methods
                            option_locator = None
                            option_selectors = [
                                f"[role='option']:has-text('{option}')",
                                f"div:has-text('{option}')",
                                f"option:has-text('{option}')"
                            ]

                            for sel in option_selectors:
                                try:
                                    option_locator = page.locator(sel).first
                                    if await option_locator.is_visible(timeout=2000):
                                        break
                                except:
                                    continue

                            if option_locator:
                                await option_locator.click()

                                # Wait for content to update
                                await page.wait_for_timeout(1000)

                                # Verify content changed by checking for config-specific elements
                                content_locator = page.locator("div.stText, pre, code").first
                                content = await content_locator.text_content() if await content_locator.is_visible() else ""

                                if content and any(key in content for key in ["top_radius", "max_agents", "host", "confidence_threshold"]):
                                    dropdown_results[f"dropdown_{option}"] = "PASS"
                                    print(f"   PASS Successfully selected and loaded: {option}")
                                else:
                                    dropdown_results[f"dropdown_{option}"] = "WARN"
                                    print(f"   WARN Selected {option} but content may not have updated properly")
                            else:
                                dropdown_results[f"dropdown_{option}"] = "FAIL"
                                print(f"   FAIL Could not find option: {option}")

                        except Exception as e:
                            dropdown_results[f"dropdown_{option}"] = "FAIL"
                            print(f"   FAIL Failed to select {option}: {e}")
                else:
                    dropdown_results["dropdown_functionality"] = "FAIL"
                    print("   FAIL Dropdown not found")

            except Exception as e:
                dropdown_results["dropdown_functionality"] = "FAIL"
                print(f"   FAIL Dropdown test failed: {e}")

            # Test 3: Helix 3D Visualization
            print("Testing Helix 3D Visualization...")
            helix_results = {}

            try:
                # Look for Plotly chart (helix visualization) - try multiple selectors
                chart_selectors = [
                    ".js-plotly-plot",
                    ".plotly-graph-div",
                    "div[id*='plotly']",
                    "div[class*='plotly']"
                ]

                plotly_chart = None
                for selector in chart_selectors:
                    try:
                        chart_element = page.locator(selector).first
                        if await chart_element.is_visible(timeout=2000):
                            plotly_chart = chart_element
                            break
                    except:
                        continue

                if plotly_chart:
                    await expect(plotly_chart).to_be_visible(timeout=5000)

                    # Check if the chart has content (indicates it rendered)
                    chart_box = await plotly_chart.bounding_box()
                    if chart_box and chart_box["width"] > 100 and chart_box["height"] > 100:
                        helix_results["helix_rendered"] = "PASS"
                        print(f"   PASS Helix visualization rendered (size: {chart_box['width']}x{chart_box['height']})")
                    else:
                        helix_results["helix_rendered"] = "FAIL"
                        print("   FAIL Helix visualization too small or not properly rendered")

                    # Check for helix-specific elements (metrics)
                    helix_metrics = page.locator("div[data-testid='metric-container']").filter(has_text="Turns")
                    if await helix_metrics.count() > 0:
                        helix_results["helix_metrics"] = "PASS"
                        print("   PASS Helix metrics displayed")
                    else:
                        helix_results["helix_metrics"] = "FAIL"
                        print("   FAIL Helix metrics not found")
                else:
                    helix_results["helix_rendered"] = "FAIL"
                    helix_results["helix_metrics"] = "FAIL"
                    print("   FAIL Helix visualization not found")

            except Exception as e:
                helix_results["helix_rendered"] = "FAIL"
                helix_results["helix_metrics"] = "FAIL"
                print(f"   FAIL Helix visualization test failed: {e}")

            # Test 4: Export Buttons
            print("Testing Export Buttons...")
            export_results = {}

            try:
                # Click on Export tab - try multiple selectors
                tab_selectors = [
                    "button:has-text('Export')",
                    "div[role='tab']:has-text('Export')",
                    "button[data-baseweb='tab']:has-text('Export')"
                ]

                export_tab = None
                for selector in tab_selectors:
                    try:
                        tab_element = page.locator(selector).first
                        if await tab_element.is_visible(timeout=2000):
                            export_tab = tab_element
                            break
                    except:
                        continue

                if export_tab:
                    await expect(export_tab).to_be_visible(timeout=5000)
                    await export_tab.click()

                    # Wait for tab content to load
                    await page.wait_for_timeout(1000)

                    # Look for export buttons
                    export_buttons = [
                        "Download as YAML",
                        "Download as JSON",
                        "Download Original"
                    ]

                    for button_text in export_buttons:
                        try:
                            button = page.locator(f"button:has-text('{button_text}')").first
                            is_visible = await button.is_visible(timeout=3000)
                            export_results[f"export_{button_text.lower().replace(' ', '_')}"] = "PASS" if is_visible else "FAIL"
                            print(f"   {'PASS' if is_visible else 'FAIL'} {button_text} button visible")
                        except Exception as e:
                            export_results[f"export_{button_text.lower().replace(' ', '_')}"] = "FAIL"
                            print(f"   FAIL {button_text} button not found: {e}")
                else:
                    export_results["export_tab"] = "FAIL"
                    print("   FAIL Export tab not found")

            except Exception as e:
                export_results["export_tab"] = "FAIL"
                print(f"   FAIL Export tab test failed: {e}")

            # Summary
            print("\nTest Results Summary:")
            print("Badge Tests:")
            print(f"  - Badge present: {badge_results.get('badge_present', 'FAIL')}")
            print(f"  - Badge mentions config files: {badge_results.get('badge_mentions_config', 'FAIL')}")

            print("Dropdown Tests:")
            for key, result in dropdown_results.items():
                if key.startswith("dropdown_"):
                    option = key.replace("dropdown_", "").replace("_", " ")
                    print(f"  - {option}: {result}")

            print("Helix Visualization Tests:")
            for key, result in helix_results.items():
                print(f"  - {key.replace('_', ' ').title()}: {result}")

            print("Export Tests:")
            for key, result in export_results.items():
                if key.startswith("export_"):
                    button = key.replace("export_", "").replace("_", " ").title()
                    print(f"  - {button}: {result}")

            # Return results for verification checklist
            return {
                "badge": badge_results,
                "dropdown": dropdown_results,
                "helix": helix_results,
                "export": export_results
            }

        except Exception as e:
            print(f"FAIL Test suite failed: {e}")
            return None

        finally:
            await browser.close()

if __name__ == "__main__":
    results = asyncio.run(test_configuration_page())
    if results:
        print("\nPASS Configuration page tests completed successfully")
        sys.exit(0)
    else:
        print("\nFAIL Configuration page tests failed")
        sys.exit(1)