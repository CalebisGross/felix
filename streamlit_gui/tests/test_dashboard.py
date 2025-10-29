import asyncio
from playwright.async_api import async_playwright

async def test_dashboard():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto('http://localhost:8501')
        await page.wait_for_load_state('networkidle')
        print("Page title:", await page.title())
        body_text = await page.locator('body').text_content()
        print("Body text:", body_text[:500])
        # Click on Dashboard in sidebar
        dashboard_link = page.get_by_role('link', name='Dashboard')
        if await dashboard_link.is_visible():
            await dashboard_link.click()
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)  # Extra wait
            print("URL after click:", page.url)
            # Wait for Dashboard specific content
            try:
                await page.wait_for_selector('text=Felix System Dashboard', timeout=10000)
                print("Dashboard content loaded")
            except:
                print("Dashboard content not loaded")
                body_text_after = await page.locator('body').text_content()
                print("Body text:", body_text_after[:500])

        results = {}

        # Check Data Source Badge
        badge = page.get_by_text('✅ Real Data')
        results['badge_present'] = await badge.is_visible()
        if results['badge_present']:
            badge_text = await badge.text_content()
            results['badge_text'] = 'This dashboard displays live metrics from Felix databases' in badge_text
        else:
            results['badge_text'] = False

        # System Metrics Tooltips
        metrics = ['System Status', 'Knowledge Entries', 'Task Patterns', 'Avg Confidence']
        for metric in metrics:
            locator = page.get_by_text(metric)
            if await locator.is_visible():
                # Check if the element has a title attribute
                title = await locator.get_attribute('title')
                results[f'{metric.lower().replace(" ", "_")}_tooltip'] = bool(title and title.strip())
            else:
                results[f'{metric.lower().replace(" ", "_")}_tooltip'] = False

        # Visual Check
        # Check for numbers in metrics
        metric_elements = page.locator('.metric, [class*="metric"]')
        count = await metric_elements.count()
        has_numbers = False
        for i in range(count):
            text = await metric_elements.nth(i).text_content()
            if any(char.isdigit() for char in text):
                has_numbers = True
                break
        results['metrics_have_numbers'] = has_numbers

        # No error messages
        error_elements = page.locator('text=/error|stack trace|exception/i')
        results['no_errors'] = await error_elements.count() == 0

        # Charts render
        chart_elements = page.locator('canvas, .plotly, [class*="chart"]')
        results['charts_present'] = await chart_elements.count() > 0

        await browser.close()
        return results

if __name__ == '__main__':
    results = asyncio.run(test_dashboard())
    print(results)