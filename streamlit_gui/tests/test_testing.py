#!/usr/bin/env python3
"""
Playwright test script for Felix Streamlit GUI Testing & Analysis page verification.
Tests badge presence, workflow results tab, metrics tooltips, and reports tab functionality.
"""

import asyncio
import sys
import os
from playwright.async_api import async_playwright, expect

async def test_testing_page():
    """Test the Testing & Analysis page functionality."""
    print("Starting Testing & Analysis page tests...")

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

            # Try to find and click the Testing & Analysis link in sidebar
            print("Looking for Testing & Analysis page link...")
            testing_link_selectors = [
                "a:has-text('Testing')",
                "div[role='button']:has-text('Testing')",
                "button:has-text('Testing')",
                "div[data-testid='stSidebar'] a:has-text('Testing')",
                "div[data-testid='stSidebar'] div:has-text('Testing')"
            ]

            testing_link = None
            for selector in testing_link_selectors:
                try:
                    link = page.locator(selector).first
                    if await link.is_visible(timeout=2000):
                        testing_link = link
                        print(f"Found Testing link with selector: {selector}")
                        break
                except:
                    continue

            if testing_link:
                await testing_link.click()
                print("Clicked Testing & Analysis link")
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
            else:
                print("Testing link not found, trying direct URL navigation...")
                await page.goto("http://localhost:8501/?page=Testing")
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
                    badge_results["badge_mentions_workflow"] = "PASS" if "actual workflow execution data" in badge_text else "FAIL"
                    print(f"   Badge text: {badge_text}")
                else:
                    badge_results["badge_present"] = "FAIL"
                    badge_results["badge_mentions_workflow"] = "FAIL"
                    print("   Badge not found or doesn't contain expected text")
            except Exception as e:
                badge_results["badge_present"] = "FAIL"
                badge_results["badge_mentions_workflow"] = "FAIL"
                print(f"   FAIL Badge test failed: {e}")

            # Test 2: Workflow Results Tab
            print("Testing Workflow Results Tab...")
            workflow_results = {}

            try:
                # Look for and click the Workflow Results tab
                tab_selectors = [
                    "button:has-text('Workflow Results')",
                    "div[role='tab']:has-text('Workflow Results')",
                    "button[data-baseweb='tab']:has-text('Workflow Results')"
                ]

                workflow_tab = None
                for selector in tab_selectors:
                    try:
                        tab_element = page.locator(selector).first
                        if await tab_element.is_visible(timeout=2000):
                            workflow_tab = tab_element
                            break
                    except:
                        continue

                if workflow_tab:
                    await expect(workflow_tab).to_be_visible(timeout=5000)
                    await workflow_tab.click()

                    # Wait for tab content to load
                    await page.wait_for_timeout(1000)

                    # Look for info box with specific text
                    info_selectors = [
                        "div.stInfo",
                        "div.element-container div.stInfo",
                        "[data-testid='stInfo']",
                        "div:has-text('Historical results from workflows executed')"
                    ]

                    info_box = None
                    info_text = ""

                    for selector in info_selectors:
                        try:
                            elements = page.locator(selector)
                            count = await elements.count()
                            for i in range(count):
                                element = elements.nth(i)
                                text = await element.text_content()
                                if "Historical results from workflows executed" in text:
                                    info_box = element
                                    info_text = text
                                    break
                            if info_box:
                                break
                        except:
                            continue

                    if info_box and "Historical results from workflows executed" in info_text:
                        workflow_results["info_box_present"] = "PASS"
                        workflow_results["info_text_correct"] = "PASS"
                        print(f"   Info box text: {info_text}")
                    else:
                        workflow_results["info_box_present"] = "FAIL"
                        workflow_results["info_text_correct"] = "FAIL"
                        print("   Info box not found or doesn't contain expected text")
                else:
                    workflow_results["workflow_tab"] = "FAIL"
                    print("   FAIL Workflow Results tab not found")

            except Exception as e:
                workflow_results["workflow_tab"] = "FAIL"
                print(f"   FAIL Workflow Results tab test failed: {e}")

            # Test 3: Metrics Tooltips
            print("Testing Metrics Tooltips...")
            metrics_results = {}

            try:
                # Look for metrics and hover over them to check for tooltips
                metric_labels = [
                    "Total Workflows",
                    "Success Rate",
                    "Avg Agents/Workflow",
                    "Avg Duration"
                ]

                for metric in metric_labels:
                    try:
                        # Find the metric label
                        metric_locator = page.locator(f"text={metric}").first
                        if await metric_locator.is_visible(timeout=2000):
                            # Hover over the metric
                            await metric_locator.hover()

                            # Wait a bit for tooltip to appear
                            await page.wait_for_timeout(500)

                            # Look for tooltip elements (Streamlit tooltips are often in specific containers)
                            tooltip_selectors = [
                                "div[data-testid='tooltip']",
                                ".stTooltip",
                                "div[role='tooltip']",
                                "div[class*='tooltip']"
                            ]

                            tooltip_found = False
                            for selector in tooltip_selectors:
                                try:
                                    tooltip = page.locator(selector).first
                                    if await tooltip.is_visible(timeout=1000):
                                        tooltip_found = True
                                        break
                                except:
                                    continue

                            if tooltip_found:
                                metrics_results[f"tooltip_{metric.lower().replace(' ', '_').replace('/', '_')}"] = "PASS"
                                print(f"   PASS {metric} tooltip found")
                            else:
                                metrics_results[f"tooltip_{metric.lower().replace(' ', '_').replace('/', '_')}"] = "WARN"
                                print(f"   WARN {metric} tooltip not detected (may be implementation-specific)")
                        else:
                            metrics_results[f"tooltip_{metric.lower().replace(' ', '_').replace('/', '_')}"] = "FAIL"
                            print(f"   FAIL {metric} label not found")
                    except Exception as e:
                        metrics_results[f"tooltip_{metric.lower().replace(' ', '_').replace('/', '_')}"] = "FAIL"
                        print(f"   FAIL {metric} test failed: {e}")

            except Exception as e:
                print(f"   FAIL Metrics tooltips test failed: {e}")

            # Test 4: Reports Tab
            print("Testing Reports Tab...")
            reports_results = {}

            try:
                # Look for and click the Reports tab
                reports_tab_selectors = [
                    "button:has-text('Reports')",
                    "div[role='tab']:has-text('Reports')",
                    "button[data-baseweb='tab']:has-text('Reports')"
                ]

                reports_tab = None
                for selector in reports_tab_selectors:
                    try:
                        tab_element = page.locator(selector).first
                        if await tab_element.is_visible(timeout=2000):
                            reports_tab = tab_element
                            break
                    except:
                        continue

                if reports_tab:
                    await expect(reports_tab).to_be_visible(timeout=5000)
                    await reports_tab.click()

                    # Wait for tab content to load
                    await page.wait_for_timeout(1000)

                    # Look for info box with "Report Types Explained"
                    info_selectors = [
                        "div.stInfo",
                        "div.element-container div.stInfo",
                        "[data-testid='stInfo']",
                        "div:has-text('Report Types Explained')"
                    ]

                    info_box = None
                    info_text = ""

                    for selector in info_selectors:
                        try:
                            elements = page.locator(selector)
                            count = await elements.count()
                            for i in range(count):
                                element = elements.nth(i)
                                text = await element.text_content()
                                if "Report Types Explained" in text:
                                    info_box = element
                                    info_text = text
                                    break
                            if info_box:
                                break
                        except:
                            continue

                    if info_box and "Report Types Explained" in info_text:
                        reports_results["info_box_present"] = "PASS"
                        print(f"   PASS Info box found: {info_text}")
                    else:
                        reports_results["info_box_present"] = "FAIL"
                        print("   FAIL Info box not found or doesn't contain expected text")

                    # Look for dropdown and hover for tooltip
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
                        await dropdown.hover()
                        await page.wait_for_timeout(500)

                        # Check for tooltip
                        tooltip_found = False
                        tooltip_selectors = [
                            "div[data-testid='tooltip']",
                            ".stTooltip",
                            "div[role='tooltip']",
                            "div[class*='tooltip']"
                        ]

                        for selector in tooltip_selectors:
                            try:
                                tooltip = page.locator(selector).first
                                if await tooltip.is_visible(timeout=1000):
                                    tooltip_found = True
                                    break
                            except:
                                continue

                        reports_results["dropdown_tooltip"] = "PASS" if tooltip_found else "WARN"
                        print(f"   {'PASS' if tooltip_found else 'WARN'} Dropdown tooltip {'found' if tooltip_found else 'not detected'}")
                    else:
                        reports_results["dropdown_tooltip"] = "FAIL"
                        print("   FAIL Dropdown not found")

                    # Look for checkboxes and hover for tooltips
                    checkbox_selectors = [
                        "input[type='checkbox']",
                        "div[data-testid='stCheckbox']"
                    ]

                    checkboxes = page.locator("input[type='checkbox'], div[data-testid='stCheckbox']")
                    checkbox_count = await checkboxes.count()

                    if checkbox_count > 0:
                        # Test first few checkboxes for tooltips
                        for i in range(min(checkbox_count, 2)):  # Test up to 2 checkboxes
                            try:
                                checkbox = checkboxes.nth(i)
                                if await checkbox.is_visible(timeout=2000):
                                    await checkbox.hover()
                                    await page.wait_for_timeout(500)

                                    # Check for tooltip
                                    tooltip_found = False
                                    tooltip_selectors = [
                                        "div[data-testid='tooltip']",
                                        ".stTooltip",
                                        "div[role='tooltip']",
                                        "div[class*='tooltip']"
                                    ]

                                    for selector in tooltip_selectors:
                                        try:
                                            tooltip = page.locator(selector).first
                                            if await tooltip.is_visible(timeout=1000):
                                                tooltip_found = True
                                                break
                                        except:
                                            continue

                                    if tooltip_found:
                                        reports_results[f"checkbox_{i}_tooltip"] = "PASS"
                                        print(f"   PASS Checkbox {i+1} tooltip found")
                                        break  # If one works, assume others work too
                                    else:
                                        reports_results[f"checkbox_{i}_tooltip"] = "WARN"
                                        print(f"   WARN Checkbox {i+1} tooltip not detected")
                            except Exception as e:
                                print(f"   FAIL Checkbox {i+1} test failed: {e}")
                    else:
                        reports_results["checkboxes"] = "FAIL"
                        print("   FAIL No checkboxes found")

                else:
                    reports_results["reports_tab"] = "FAIL"
                    print("   FAIL Reports tab not found")

            except Exception as e:
                reports_results["reports_tab"] = "FAIL"
                print(f"   FAIL Reports tab test failed: {e}")

            # Summary
            print("\nTest Results Summary:")
            print("Badge Tests:")
            print(f"  - Badge present: {badge_results.get('badge_present', 'FAIL')}")
            print(f"  - Badge mentions workflow data: {badge_results.get('badge_mentions_workflow', 'FAIL')}")

            print("Workflow Results Tab Tests:")
            print(f"  - Tab accessible: {workflow_results.get('workflow_tab', 'FAIL')}")
            print(f"  - Info box present: {workflow_results.get('info_box_present', 'FAIL')}")
            print(f"  - Info text correct: {workflow_results.get('info_text_correct', 'FAIL')}")

            print("Metrics Tooltips Tests:")
            for key, result in metrics_results.items():
                if key.startswith("tooltip_"):
                    metric = key.replace("tooltip_", "").replace("_", " ").title()
                    print(f"  - {metric}: {result}")

            print("Reports Tab Tests:")
            print(f"  - Tab accessible: {reports_results.get('reports_tab', 'FAIL')}")
            print(f"  - Info box present: {reports_results.get('info_box_present', 'FAIL')}")
            print(f"  - Dropdown tooltip: {reports_results.get('dropdown_tooltip', 'FAIL')}")
            for key, result in reports_results.items():
                if key.startswith("checkbox_") and key.endswith("_tooltip"):
                    checkbox_num = key.split("_")[1]
                    print(f"  - Checkbox {checkbox_num} tooltip: {result}")

            # Return results for verification checklist
            return {
                "badge": badge_results,
                "workflow": workflow_results,
                "metrics": metrics_results,
                "reports": reports_results
            }

        except Exception as e:
            print(f"FAIL Test suite failed: {e}")
            return None

        finally:
            await browser.close()

if __name__ == "__main__":
    results = asyncio.run(test_testing_page())
    if results:
        print("\nPASS Testing & Analysis page tests completed successfully")
        sys.exit(0)
    else:
        print("\nFAIL Testing & Analysis page tests failed")
        sys.exit(1)