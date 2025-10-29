from playwright.sync_api import sync_playwright

def test_streamlit_errors():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        errors = []

        def on_console(msg):
            if msg.type == 'error':
                errors.append(f"Console error: {msg.text}")
                print(f"Console error: {msg.text}")

        def on_request_failed(req):
            errors.append(f"Failed request: {req.url} - {req.failure}")
            print(f"Failed request: {req.url} - {req.failure}")

        page.on('console', on_console)
        page.on('requestfailed', on_request_failed)

        # Wait for sidebar to load
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=10000)

        # Navigate to Configuration
        config_link = page.locator('[data-testid="stSidebar"]').locator('text=Configuration')
        if config_link.is_visible(timeout=5000):
            config_link.click()
            page.wait_for_timeout(1000)

        # Navigate to Testing
        testing_link = page.locator('[data-testid="stSidebar"]').locator('text=Testing')
        if testing_link.is_visible(timeout=5000):
            testing_link.click()
            page.wait_for_timeout(1000)

        # Navigate to Benchmarking
        benchmarking_link = page.locator('[data-testid="stSidebar"]').locator('text=Benchmarking')
        if benchmarking_link.is_visible(timeout=5000):
            benchmarking_link.click()
            page.wait_for_timeout(1000)

        # Back to Dashboard
        dashboard_link = page.locator('[data-testid="stSidebar"]').locator('text=Dashboard')
        if dashboard_link.is_visible(timeout=5000):
            dashboard_link.click()
            page.wait_for_timeout(1000)

        browser.close()

        if errors:
            print("Errors found:")
            for error in errors:
                print(error)
            raise AssertionError("Errors detected in browser console or network requests")
        else:
            print("No errors found in browser console or network requests")

if __name__ == "__main__":
    test_streamlit_errors()