#!/usr/bin/env python3
"""
Comprehensive Playwright test for Felix Streamlit GUI - Benchmarking Page
Tests all items in the verification checklist for Page 4: Benchmarking
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from playwright.async_api import async_playwright, expect, Page, Browser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class BenchmarkingTester:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()

    async def test_benchmarking_page(self):
        """Main test function for Benchmarking page"""
        print("[ROCKET] Starting Benchmarking Page Tests...")

        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Navigate to Benchmarking page
                print("[LOCATION] Navigating to Benchmarking page...")
                await page.goto("http://localhost:8501/?page=Benchmarking")
                await page.wait_for_load_state("networkidle")

                # Wait for page to load - try multiple possible selectors
                try:
                    await expect(page.locator("h1:has-text('Performance Benchmarking')")).to_be_visible(timeout=10000)
                except:
                    try:
                        await expect(page.locator("h1").first).to_be_visible(timeout=5000)
                    except:
                        await expect(page.locator("text=Benchmarking")).to_be_visible(timeout=5000)

                print("[OK] Page loaded successfully")

                # Test 1: Mode Selector
                await self.test_mode_selector(page)

                # Test 2: Data Source Badges
                await self.test_data_source_badges(page)

                # Test 3: Hypothesis Explanations
                await self.test_hypothesis_explanations(page)

                # Test 4: Configuration Section
                await self.test_configuration_section(page)

                # Test 5: Demo Mode Test
                await self.test_demo_mode(page)

                # Test 6: Real Mode Test
                await self.test_real_mode(page)

                # Test 7: Performance Tests Tab
                await self.test_performance_tests_tab(page)

                # Take final screenshot
                await page.screenshot(path="benchmarking_final.png", full_page=True)
                print("[CAMERA] Final screenshot saved")

            except Exception as e:
                print(f"[FAIL] Test failed with error: {e}")
                self.results['error'] = str(e)

            finally:
                await browser.close()

        return self.results

    async def test_mode_selector(self, page: Page):
        """Test Mode Selector functionality"""
        print("\n[TEST] Testing Mode Selector...")

        # Check radio buttons exist
        demo_radio = page.locator("input[type='radio'][value*='Demo']")
        real_radio = page.locator("input[type='radio'][value*='Real']")

        if await demo_radio.is_visible() and await real_radio.is_visible():
            self.results['mode_selector_radios'] = "[OK]"
            print("  [OK] Radio buttons appear")
        else:
            self.results['mode_selector_radios'] = "[FAIL]"
            print("  [FAIL] Radio buttons not found")

        # Check default selection (Demo Mode)
        demo_checked = await demo_radio.is_checked()
        if demo_checked:
            self.results['mode_selector_default'] = "[OK]"
            print("  [OK] Default selection is Demo Mode")
        else:
            self.results['mode_selector_default'] = "[FAIL]"
            print("  [FAIL] Default selection is not Demo Mode")

        # Check availability indicator
        try:
            # Look for success/warning indicators in the second column
            mode_col2 = page.locator(".stColumn").nth(1)
            indicator_text = await mode_col2.inner_text()
            if "[OK]" in indicator_text or "[WARN]" in indicator_text:
                self.results['mode_selector_indicator'] = "[OK]"
                print(f"  [OK] Availability indicator shows: {indicator_text.strip()}")
            else:
                self.results['mode_selector_indicator'] = "[FAIL]"
                print("  [FAIL] No availability indicator found")
        except:
            self.results['mode_selector_indicator'] = "[FAIL]"
            print("  [FAIL] Could not find availability indicator")

        # Test tooltip on radio buttons
        try:
            await demo_radio.hover()
            await page.wait_for_timeout(500)
            tooltip = page.locator(".stTooltip")
            tooltip_visible = await tooltip.is_visible()
            if tooltip_visible:
                self.results['mode_selector_tooltip'] = "[OK]"
                print("  [OK] Tooltip appears on hover")
            else:
                self.results['mode_selector_tooltip'] = "[FAIL]"
                print("  [FAIL] No tooltip on hover")
        except:
            self.results['mode_selector_tooltip'] = "[FAIL]"
            print("  [FAIL] Could not test tooltip")

    async def test_data_source_badges(self, page: Page):
        """Test Data Source Badges for both modes"""
        print("\n[TEST] Testing Data Source Badges...")

        # Test Demo Mode badge
        demo_radio = page.locator("input[type='radio'][value*='Demo']")
        await demo_radio.check()

        try:
            # Look for warning box with demo mode text
            warning_box = page.locator("text=[WARN] Demo Mode").or_(page.locator("text=Demo Mode"))
            if await warning_box.is_visible():
                warning_text = await warning_box.inner_text()
                if "Demo Mode" in warning_text and ("statistical models" in warning_text.lower() or "simulated" in warning_text.lower()):
                    self.results['demo_badge'] = "[OK]"
                    print("  [OK] Demo Mode badge shows correct warning text")
                else:
                    self.results['demo_badge'] = "[WARN]"
                    print(f"  [WARN] Demo Mode badge visible but text unclear: {warning_text}")
            else:
                self.results['demo_badge'] = "[FAIL]"
                print("  [FAIL] Demo Mode badge not found")
        except Exception as e:
            self.results['demo_badge'] = "[FAIL]"
            print(f"  [FAIL] Error testing demo badge: {e}")

        # Test Real Mode badge
        real_radio = page.locator("input[type='radio'][value*='Real']")
        await real_radio.check()

        try:
            # Look for success or error box
            real_indicators = page.locator("text=[OK] Real Benchmarks").or_(page.locator("text=[FAIL] Real mode unavailable"))
            if await real_indicators.is_visible():
                real_text = await real_indicators.inner_text()
                if "Real" in real_text:
                    self.results['real_badge'] = "[OK]"
                    print(f"  [OK] Real Mode badge shows: {real_text.strip()}")
                    if "unavailable" in real_text.lower():
                        self.results['real_badge_note'] = "Falls back to simulated"
                        print("    ℹ️ Real mode unavailable, falls back to simulated")
                else:
                    self.results['real_badge'] = "[FAIL]"
                    print("  [FAIL] Real Mode badge text unclear")
            else:
                self.results['real_badge'] = "[FAIL]"
                print("  [FAIL] Real Mode badge not found")
        except Exception as e:
            self.results['real_badge'] = "[FAIL]"
            print(f"  [FAIL] Error testing real badge: {e}")

    async def test_hypothesis_explanations(self, page: Page):
        """Test Hypothesis Explanations"""
        print("\n[TEST] Testing Hypothesis Explanations...")

        hypotheses = [
            ("H1", "Helical Progression"),
            ("H2", "Hub-Spoke Communication"),
            ("H3", "Memory Compression")
        ]

        for h_id, h_name in hypotheses:
            try:
                # Find and click the expander
                expander = page.locator(f"text=ℹ️ **{h_id}: {h_name}**").or_(page.locator(f"text={h_id}: {h_name}"))
                if await expander.is_visible():
                    await expander.click()

                    # Wait for content to expand
                    await page.wait_for_timeout(1000)

                    # Check for key content sections
                    content_checks = [
                        "What it tests",
                        "How it works",
                        "Measured by",
                        "Why"
                    ]

                    content_found = []
                    for check in content_checks:
                        if await page.locator(f"text={check}").is_visible():
                            content_found.append(check)

                    if len(content_found) >= 3:  # At least 3 out of 4 sections
                        self.results[f'hypothesis_{h_id.lower()}'] = "[OK]"
                        print(f"  [OK] {h_id} explanation expands and shows detailed content")
                    else:
                        self.results[f'hypothesis_{h_id.lower()}'] = "[WARN]"
                        print(f"  [WARN] {h_id} explanation expands but missing some content sections")
                else:
                    self.results[f'hypothesis_{h_id.lower()}'] = "[FAIL]"
                    print(f"  [FAIL] {h_id} explanation not found")

            except Exception as e:
                self.results[f'hypothesis_{h_id.lower()}'] = "[FAIL]"
                print(f"  [FAIL] Error testing {h_id}: {e}")

    async def test_configuration_section(self, page: Page):
        """Test Configuration Section"""
        print("\n[TEST] Testing Configuration Section...")

        # Check info box
        try:
            info_box = page.locator("text=Tip").or_(page.locator("text=Tip: Larger sample sizes"))
            if await info_box.is_visible():
                info_text = await info_box.inner_text()
                if "sample sizes" in info_text.lower() and "500" in info_text:
                    self.results['config_info_box'] = "[OK]"
                    print("  [OK] Info box appears with tip about sample sizes")
                else:
                    self.results['config_info_box'] = "[WARN]"
                    print(f"  [WARN] Info box visible but text unclear: {info_text}")
            else:
                self.results['config_info_box'] = "[FAIL]"
                print("  [FAIL] Info box not found")
        except Exception as e:
            self.results['config_info_box'] = "[FAIL]"
            print(f"  [FAIL] Error testing info box: {e}")

        # Test checkbox tooltips
        checkboxes = [
            ("Test H1", "input[type='checkbox']", 0),
            ("Test H2", "input[type='checkbox']", 1),
            ("Test H3", "input[type='checkbox']", 2)
        ]

        for cb_name, selector, index in checkboxes:
            try:
                checkbox = page.locator(selector).nth(index)
                if await checkbox.is_visible():
                    await checkbox.hover()
                    await page.wait_for_timeout(500)
                    tooltip = page.locator(".stTooltip")
                    tooltip_visible = await tooltip.is_visible()
                    if tooltip_visible:
                        self.results[f'checkbox_tooltip_{cb_name.replace(" ", "_").lower()}'] = "[OK]"
                        print(f"  [OK] {cb_name} checkbox tooltip appears")
                    else:
                        self.results[f'checkbox_tooltip_{cb_name.replace(" ", "_").lower()}'] = "[FAIL]"
                        print(f"  [FAIL] {cb_name} checkbox tooltip not found")
                else:
                    self.results[f'checkbox_tooltip_{cb_name.replace(" ", "_").lower()}'] = "[FAIL]"
                    print(f"  [FAIL] {cb_name} checkbox not found")
            except Exception as e:
                self.results[f'checkbox_tooltip_{cb_name.replace(" ", "_").lower()}'] = "[FAIL]"
                print(f"  [FAIL] Error testing {cb_name} checkbox: {e}")

        # Test sample size tooltips
        sample_inputs = [
            ("H1", "input[type='number']", 0),
            ("H2", "input[type='number']", 1),
            ("H3", "input[type='number']", 2)
        ]

        for sample_name, selector, index in sample_inputs:
            try:
                sample_input = page.locator(selector).nth(index)
                if await sample_input.is_visible():
                    await sample_input.hover()
                    await page.wait_for_timeout(500)
                    tooltip = page.locator(".stTooltip")
                    tooltip_visible = await tooltip.is_visible()
                    if tooltip_visible:
                        self.results[f'sample_tooltip_{sample_name.lower()}'] = "[OK]"
                        print(f"  [OK] {sample_name} sample size tooltip appears")
                    else:
                        self.results[f'sample_tooltip_{sample_name.lower()}'] = "[FAIL]"
                        print(f"  [FAIL] {sample_name} sample size tooltip not found")
                else:
                    self.results[f'sample_tooltip_{sample_name.lower()}'] = "[FAIL]"
                    print(f"  [FAIL] {sample_name} sample size input not found")
            except Exception as e:
                self.results[f'sample_tooltip_{sample_name.lower()}'] = "[FAIL]"
                print(f"  [FAIL] Error testing {sample_name} sample input: {e}")

    async def test_demo_mode(self, page: Page):
        """Test Demo Mode execution"""
        print("\n[TEST] Testing Demo Mode...")

        # Select Demo Mode
        demo_radio = page.locator("input[type='radio'][value*='Demo']")
        await demo_radio.check()

        # Check all hypotheses
        await page.locator("input[type='checkbox']").first.check()
        await page.locator("input[type='checkbox']").nth(1).check()
        await page.locator("input[type='checkbox']").nth(2).check()

        # Set sample sizes to 50
        sample_inputs = page.locator("input[type='number']")
        for i in range(await sample_inputs.count()):
            await sample_inputs.nth(i).fill("50")

        # Click run button
        run_button = page.locator("button:has-text('Run Hypothesis Validation')")
        if await run_button.is_visible():
            await run_button.click()
            print("  [OK] Run button clicked")

            # Wait for spinner
            try:
                spinner = page.locator(".stSpinner").or_(page.locator("text=Running benchmarks"))
                await expect(spinner).to_be_visible(timeout=5000)
                self.results['demo_spinner'] = "[OK]"
                print("  [OK] Spinner appears")
            except:
                self.results['demo_spinner'] = "[FAIL]"
                print("  [FAIL] Spinner not found")

            # Check info message
            try:
                info_msg = page.locator("text=Running DEMO benchmarks").or_(page.locator("text=DEMO benchmarks"))
                await expect(info_msg).to_be_visible(timeout=10000)
                self.results['demo_info_message'] = "[OK]"
                print("  [OK] Demo info message appears")
            except:
                self.results['demo_info_message'] = "[FAIL]"
                print("  [FAIL] Demo info message not found")

            # Wait for completion and check success message
            try:
                success_msg = page.locator("text=[OK] Demo benchmark completed").or_(page.locator("text=Demo benchmark completed"))
                await expect(success_msg).to_be_visible(timeout=30000)
                self.results['demo_success'] = "[OK]"
                print("  [OK] Demo success message appears")
            except:
                self.results['demo_success'] = "[FAIL]"
                print("  [FAIL] Demo success message not found")

            # Check badge
            try:
                badge = page.locator("text=Data Source: SIMULATED").or_(page.locator("text=SIMULATED"))
                if await badge.is_visible():
                    self.results['demo_badge_simulated'] = "[OK]"
                    print("  [OK] Simulated badge appears")
                else:
                    self.results['demo_badge_simulated'] = "[FAIL]"
                    print("  [FAIL] Simulated badge not found")
            except:
                self.results['demo_badge_simulated'] = "[FAIL]"
                print("  [FAIL] Error checking simulated badge")

            # Check results display
            try:
                results_section = page.locator("text=Validation Results").or_(page.locator("text=Results"))
                if await results_section.is_visible():
                    self.results['demo_results'] = "[OK]"
                    print("  [OK] Results display appears")
                else:
                    self.results['demo_results'] = "[FAIL]"
                    print("  [FAIL] Results display not found")
            except:
                self.results['demo_results'] = "[FAIL]"
                print("  [FAIL] Error checking results display")

            # Check chart
            try:
                chart = page.locator(".js-plotly-plot").or_(page.locator("text=Hypothesis Validation Results"))
                if await chart.is_visible():
                    self.results['demo_chart'] = "[OK]"
                    print("  [OK] Chart appears")
                else:
                    self.results['demo_chart'] = "[FAIL]"
                    print("  [FAIL] Chart not found")
            except:
                self.results['demo_chart'] = "[FAIL]"
                print("  [FAIL] Error checking chart")

            # Test detailed statistics expansion
            try:
                details_expander = page.locator("text=[CHART] Detailed Statistics").or_(page.locator("text=Detailed Statistics"))
                if await details_expander.is_visible():
                    await details_expander.click()
                    await page.wait_for_timeout(1000)

                    # Check for baseline/treatment means
                    baseline = page.locator("text=Baseline Mean").or_(page.locator("text=Baseline:"))
                    treatment = page.locator("text=Treatment Mean").or_(page.locator("text=Treatment:"))
                    gain = page.locator("text=Actual Gain").or_(page.locator("text=Gain"))

                    if await baseline.is_visible() and await treatment.is_visible() and await gain.is_visible():
                        self.results['demo_detailed_stats'] = "[OK]"
                        print("  [OK] Detailed statistics expand and show baseline/treatment/gain")
                    else:
                        self.results['demo_detailed_stats'] = "[WARN]"
                        print("  [WARN] Detailed statistics expand but missing some metrics")
                else:
                    self.results['demo_detailed_stats'] = "[FAIL]"
                    print("  [FAIL] Detailed statistics expander not found")
            except Exception as e:
                self.results['demo_detailed_stats'] = "[FAIL]"
                print(f"  [FAIL] Error testing detailed statistics: {e}")

        else:
            print("  [FAIL] Run button not found")
            self.results['demo_run_button'] = "[FAIL]"

    async def test_real_mode(self, page: Page):
        """Test Real Mode execution"""
        print("\n[TEST] Testing Real Mode...")

        # Select Real Mode
        real_radio = page.locator("input[type='radio'][value*='Real']")
        await real_radio.check()

        # Check all hypotheses
        await page.locator("input[type='checkbox']").first.check()
        await page.locator("input[type='checkbox']").nth(1).check()
        await page.locator("input[type='checkbox']").nth(2).check()

        # Set sample sizes to 20
        sample_inputs = page.locator("input[type='number']")
        for i in range(await sample_inputs.count()):
            await sample_inputs.nth(i).fill("20")

        # Click run button
        run_button = page.locator("button:has-text('Run Hypothesis Validation')")
        if await run_button.is_visible():
            await run_button.click()
            print("  [OK] Run button clicked")

            # Wait for spinner
            try:
                spinner = page.locator(".stSpinner").or_(page.locator("text=Running benchmarks"))
                await expect(spinner).to_be_visible(timeout=5000)
                self.results['real_spinner'] = "[OK]"
                print("  [OK] Spinner appears")
            except:
                self.results['real_spinner'] = "[FAIL]"
                print("  [FAIL] Spinner not found")

            # Check info message
            try:
                info_msg = page.locator("text=Running REAL benchmarks").or_(page.locator("text=REAL benchmarks"))
                await expect(info_msg).to_be_visible(timeout=10000)
                self.results['real_info_message'] = "[OK]"
                print("  [OK] Real info message appears")
            except:
                self.results['real_info_message'] = "[FAIL]"
                print("  [FAIL] Real info message not found")

            # Check for individual spinners (H1, H2, H3)
            try:
                individual_spinners = page.locator("text=Testing H1").or_(page.locator("text=Testing H2").or_(page.locator("text=Testing H3")))
                spinner_count = 0
                for i in range(3):
                    if await individual_spinners.nth(i).is_visible():
                        spinner_count += 1

                if spinner_count >= 1:  # At least one individual spinner
                    self.results['real_individual_spinners'] = "[OK]"
                    print(f"  [OK] Individual spinners appear ({spinner_count}/3 found)")
                else:
                    self.results['real_individual_spinners'] = "[FAIL]"
                    print("  [FAIL] No individual spinners found")
            except:
                self.results['real_individual_spinners'] = "[FAIL]"
                print("  [FAIL] Error checking individual spinners")

            # Wait for completion and check success message
            try:
                success_msg = page.locator("text=[OK] Real benchmark completed").or_(page.locator("text=Real benchmark completed"))
                await expect(success_msg).to_be_visible(timeout=60000)  # Longer timeout for real mode
                self.results['real_success'] = "[OK]"
                print("  [OK] Real success message appears")
            except:
                self.results['real_success'] = "[FAIL]"
                print("  [FAIL] Real success message not found")

            # Check badge (could be REAL or SIMULATED with reason)
            try:
                real_badge = page.locator("text=Data Source: REAL").or_(page.locator("text=REAL"))
                simulated_badge = page.locator("text=Data Source: SIMULATED").or_(page.locator("text=SIMULATED"))

                if await real_badge.is_visible():
                    self.results['real_badge_final'] = "[OK]"
                    self.results['real_badge_note'] = "Uses REAL components"
                    print("  [OK] Real badge shows: Uses actual Felix components")
                elif await simulated_badge.is_visible():
                    badge_text = await simulated_badge.inner_text()
                    self.results['real_badge_final'] = "[WARN]"
                    self.results['real_badge_note'] = "Fell back to simulated"
                    print(f"  [WARN] Real badge shows fallback: {badge_text}")
                else:
                    self.results['real_badge_final'] = "[FAIL]"
                    print("  [FAIL] No badge found after real mode")
            except:
                self.results['real_badge_final'] = "[FAIL]"
                print("  [FAIL] Error checking final badge")

            # Check individual hypothesis sources
            try:
                h1_source = page.locator("text=H1").locator("..").locator("text=[OK]").or_(page.locator("text=H1").locator("..").locator("text=SIMULATED"))
                h2_source = page.locator("text=H2").locator("..").locator("text=[OK]").or_(page.locator("text=H2").locator("..").locator("text=SIMULATED"))
                h3_source = page.locator("text=H3").locator("..").locator("text=[OK]").or_(page.locator("text=H3").locator("..").locator("text=SIMULATED"))

                source_indicators = 0
                for source in [h1_source, h2_source, h3_source]:
                    if await source.is_visible():
                        source_indicators += 1

                if source_indicators >= 1:
                    self.results['real_individual_sources'] = "[OK]"
                    print(f"  [OK] Individual hypothesis sources shown ({source_indicators}/3)")
                else:
                    self.results['real_individual_sources'] = "[FAIL]"
                    print("  [FAIL] No individual hypothesis sources found")
            except:
                self.results['real_individual_sources'] = "[FAIL]"
                print("  [FAIL] Error checking individual sources")

            # Check results display
            try:
                results_section = page.locator("text=Validation Results").or_(page.locator("text=Results"))
                if await results_section.is_visible():
                    self.results['real_results'] = "[OK]"
                    print("  [OK] Results display appears")
                else:
                    self.results['real_results'] = "[FAIL]"
                    print("  [FAIL] Results display not found")
            except:
                self.results['real_results'] = "[FAIL]"
                print("  [FAIL] Error checking results display")

            # Check chart
            try:
                chart = page.locator(".js-plotly-plot").or_(page.locator("text=Hypothesis Validation Results"))
                if await chart.is_visible():
                    self.results['real_chart'] = "[OK]"
                    print("  [OK] Chart appears")
                else:
                    self.results['real_chart'] = "[FAIL]"
                    print("  [FAIL] Chart not found")
            except:
                self.results['real_chart'] = "[FAIL]"
                print("  [FAIL] Error checking chart")

            # Test detailed statistics expansion
            try:
                details_expander = page.locator("text=[CHART] Detailed Statistics").or_(page.locator("text=Detailed Statistics"))
                if await details_expander.is_visible():
                    await details_expander.click()
                    await page.wait_for_timeout(1000)

                    # Check for baseline/treatment means
                    baseline = page.locator("text=Baseline Mean").or_(page.locator("text=Baseline:"))
                    treatment = page.locator("text=Treatment Mean").or_(page.locator("text=Treatment:"))
                    gain = page.locator("text=Actual Gain").or_(page.locator("text=Gain"))

                    if await baseline.is_visible() and await treatment.is_visible() and await gain.is_visible():
                        self.results['real_detailed_stats'] = "[OK]"
                        print("  [OK] Detailed statistics expand and show baseline/treatment/gain")
                    else:
                        self.results['real_detailed_stats'] = "[WARN]"
                        print("  [WARN] Detailed statistics expand but missing some metrics")
                else:
                    self.results['real_detailed_stats'] = "[FAIL]"
                    print("  [FAIL] Detailed statistics expander not found")
            except Exception as e:
                self.results['real_detailed_stats'] = "[FAIL]"
                print(f"  [FAIL] Error testing detailed statistics: {e}")

        else:
            print("  [FAIL] Run button not found")
            self.results['real_run_button'] = "[FAIL]"

    async def test_performance_tests_tab(self, page: Page):
        """Test Performance Tests Tab"""
        print("\n[TEST] Testing Performance Tests Tab...")

        # Switch to Performance Tests tab
        try:
            perf_tab = page.locator("text=⚡ Performance Tests").or_(page.locator("text=Performance Tests"))
            await perf_tab.click()
            await page.wait_for_timeout(2000)
            print("  [OK] Switched to Performance Tests tab")
        except:
            print("  [FAIL] Could not switch to Performance Tests tab")
            return

        # Check yellow warning
        try:
            warning = page.locator("text=[WARN] Note").or_(page.locator("text=Note: Performance tests currently use simulated data"))
            if await warning.is_visible():
                warning_text = await warning.inner_text()
                if "simulated data" in warning_text.lower():
                    self.results['perf_tests_warning'] = "[OK]"
                    print("  [OK] Yellow warning about simulated data appears")
                else:
                    self.results['perf_tests_warning'] = "[WARN]"
                    print(f"  [WARN] Warning visible but text unclear: {warning_text}")
            else:
                self.results['perf_tests_warning'] = "[FAIL]"
                print("  [FAIL] Warning not found")
        except Exception as e:
            self.results['perf_tests_warning'] = "[FAIL]"
            print(f"  [FAIL] Error testing warning: {e}")

        # Check test categories explanation
        try:
            categories_section = page.locator("text=Test Categories").or_(page.locator("text=Each category tests"))
            if await categories_section.is_visible():
                categories_text = await categories_section.inner_text()
                if "tests a specific aspect" in categories_text.lower():
                    self.results['perf_tests_explanation'] = "[OK]"
                    print("  [OK] Test categories explanation appears")
                else:
                    self.results['perf_tests_explanation'] = "[WARN]"
                    print(f"  [WARN] Categories section visible but text unclear: {categories_text}")
            else:
                self.results['perf_tests_explanation'] = "[FAIL]"
                print("  [FAIL] Test categories explanation not found")
        except Exception as e:
            self.results['perf_tests_explanation'] = "[FAIL]"
            print(f"  [FAIL] Error testing categories explanation: {e}")

        # Test dropdown selection
        try:
            dropdown = page.locator("select").or_(page.locator(".stSelectbox"))
            if await dropdown.is_visible():
                # Try to select a different option
                await dropdown.click()
                await page.wait_for_timeout(1000)

                # Check if options are available
                options = dropdown.locator("option")
                option_count = await options.count()
                if option_count > 1:
                    self.results['perf_tests_dropdown'] = "[OK]"
                    print(f"  [OK] Dropdown works with {option_count} options")
                else:
                    self.results['perf_tests_dropdown'] = "[WARN]"
                    print("  [WARN] Dropdown visible but no options found")
            else:
                self.results['perf_tests_dropdown'] = "[FAIL]"
                print("  [FAIL] Dropdown not found")
        except Exception as e:
            self.results['perf_tests_dropdown'] = "[FAIL]"
            print(f"  [FAIL] Error testing dropdown: {e}")

        # Check info box for selected test
        try:
            info_box = page.locator("text=**Agent Spawning**").or_(page.locator("text=**Message Routing**").or_(page.locator("text=**Memory Operations**")))
            if await info_box.is_visible():
                self.results['perf_tests_info_box'] = "[OK]"
                print("  [OK] Info box appears for selected test")
            else:
                self.results['perf_tests_info_box'] = "[FAIL]"
                print("  [FAIL] Info box not found")
        except Exception as e:
            self.results['perf_tests_info_box'] = "[FAIL]"
            print(f"  [FAIL] Error testing info box: {e}")

        # Test dropdown tooltip
        try:
            if 'dropdown' in locals() and await dropdown.is_visible():
                await dropdown.hover()
                await page.wait_for_timeout(500)
                tooltip = page.locator(".stTooltip")
                tooltip_visible = await tooltip.is_visible()
                if tooltip_visible:
                    self.results['perf_tests_dropdown_tooltip'] = "[OK]"
                    print("  [OK] Dropdown tooltip appears")
                else:
                    self.results['perf_tests_dropdown_tooltip'] = "[FAIL]"
                    print("  [FAIL] Dropdown tooltip not found")
            else:
                self.results['perf_tests_dropdown_tooltip'] = "[FAIL]"
                print("  [FAIL] Cannot test tooltip - dropdown not found")
        except Exception as e:
            self.results['perf_tests_dropdown_tooltip'] = "[FAIL]"
            print(f"  [FAIL] Error testing dropdown tooltip: {e}")


async def main():
    """Main function to run all tests"""
    tester = BenchmarkingTester()

    print("[MICROSCOPE] Felix Streamlit GUI - Benchmarking Page Verification")
    print("=" * 60)

    try:
        results = await tester.test_benchmarking_page()

        # Print summary
        print("\n" + "=" * 60)
        print("[CHART] TEST RESULTS SUMMARY")
        print("=" * 60)

        # Count results by status
        success_count = sum(1 for r in results.values() if r == "[OK]")
        warning_count = sum(1 for r in results.values() if r == "[WARN]")
        error_count = sum(1 for r in results.values() if r == "[FAIL]")
        total_count = len(results)

        print(f"[OK] Success: {success_count}")
        print(f"[WARN]  Warning: {warning_count}")
        print(f"[FAIL] Error: {error_count}")
        print(f"[CHART] Total: {total_count}")

        if error_count == 0:
            print("\n[PARTY] All tests passed!")
        elif success_count > error_count:
            print(f"\nMostly working ({success_count}/{total_count} passed)")
        else:
            print(f"\n[WARN]  Issues found ({error_count}/{total_count} failed)")

        # Save detailed results
        end_time = datetime.now()
        results['test_duration'] = str(end_time - tester.start_time)
        results['timestamp'] = end_time.isoformat()

        with open('benchmarking_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("[SAVE] Detailed results saved to benchmarking_test_results.json")

        return results

    except Exception as e:
        print(f"[BOMB] Test suite failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    results = asyncio.run(main())
    sys.exit(0 if 'error' not in results else 1)