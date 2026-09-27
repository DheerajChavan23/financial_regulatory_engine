"""Capture authentic high-resolution DARK THEME screenshots of the running Streamlit dashboard using Selenium."""

from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def capture_all_tabs_dark() -> None:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1680,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--force-dark-mode")
    options.add_argument("--blink-settings=forceDarkModeEnabled=true")

    driver = webdriver.Chrome(options=options)
    dst_dir = Path("docs/images")
    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Enforce dark color scheme preference via Chrome DevTools Protocol
        driver.execute_cdp_cmd(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-color-scheme", "value": "dark"}]},
        )

        url = "http://localhost:8501"
        driver.get(url)
        time.sleep(5)

        # 1. Click Tab 4: Executive BI & Analytics Mart
        bi_tab_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Executive BI')]")
        driver.execute_script("arguments[0].click();", bi_tab_elem)
        time.sleep(5)  # Wait for Plotly dark charts to render

        bi_screenshot = dst_dir / "streamlit_bi_dashboard.png"
        driver.save_screenshot(str(bi_screenshot.resolve()))
        print(f"Captured Dark Theme Tab 4 (Executive BI): {bi_screenshot.name} ({bi_screenshot.stat().st_size} bytes)")

        # 2. Click Tab 1: Ingestion Pipeline
        pipe_tab_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Ingestion Pipeline')]")
        driver.execute_script("arguments[0].click();", pipe_tab_elem)
        time.sleep(4)

        pipe_screenshot = dst_dir / "streamlit_ingestion_pipeline.png"
        driver.save_screenshot(str(pipe_screenshot.resolve()))
        print(f"Captured Dark Theme Tab 1 (Pipeline): {pipe_screenshot.name} ({pipe_screenshot.stat().st_size} bytes)")

        # 3. Click Tab 2: Human-in-the-Loop Quarantine Review
        q_tab_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Human-in-the-Loop')]")
        driver.execute_script("arguments[0].click();", q_tab_elem)
        time.sleep(4)

        q_screenshot = dst_dir / "streamlit_quarantine_review.png"
        driver.save_screenshot(str(q_screenshot.resolve()))
        print(f"Captured Dark Theme Tab 2 (Quarantine Review): {q_screenshot.name} ({q_screenshot.stat().st_size} bytes)")

    finally:
        driver.quit()


if __name__ == "__main__":
    capture_all_tabs_dark()
