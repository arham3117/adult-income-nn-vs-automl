"""Step 5 (part 5) - Capture screenshots of the RUNNING deployment.

Drives a headless Chromium against the live uvicorn server, so every image is a
real render of the service while it is serving, not a mock-up.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

BASE = "http://127.0.0.1:8000"
SHOTS = C.REPORTS / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    requests.get(f"{BASE}/health", timeout=15).raise_for_status()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        # 1. Swagger UI landing page
        page.goto(f"{BASE}/docs", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SHOTS / "01_swagger_docs.png"), full_page=True)
        print("  01_swagger_docs.png")

        # 2. /health response
        page.goto(f"{BASE}/health", wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOTS / "02_health_endpoint.png"))
        print("  02_health_endpoint.png")

        # 3. Execute a real /predict call from inside Swagger UI
        page.goto(f"{BASE}/docs", wait_until="networkidle")
        page.wait_for_timeout(1200)
        page.click("#operations-default-predict_one_predict_post")
        page.wait_for_timeout(600)
        page.click("button.try-out__btn")
        page.wait_for_timeout(400)
        body = {
            "age": 52, "workclass": "Self-emp-inc", "education-num": 14,
            "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
            "relationship": "Husband", "race": "White", "sex": "Male",
            "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
            "native-country": "United-States",
        }
        page.evaluate(
            """(payload) => {
                const ta = document.querySelector('.body-param__text');
                if (ta) {
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(ta, JSON.stringify(payload, null, 2));
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }""", body)
        page.wait_for_timeout(500)
        page.click("button.execute")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SHOTS / "03_swagger_predict_executed.png"), full_page=True)
        print("  03_swagger_predict_executed.png")

        # 4. The unseen-category request, executed live
        page.goto(f"{BASE}/docs", wait_until="networkidle")
        page.wait_for_timeout(1200)
        page.click("#operations-default-predict_one_predict_post")
        page.wait_for_timeout(600)
        page.click("button.try-out__btn")
        page.wait_for_timeout(400)
        unseen = {**body, "age": 37, "workclass": "Private", "education-num": 13,
                  "occupation": "Tech-support", "race": "Black",
                  "capital-gain": 0, "hours-per-week": 40, "native-country": "Wakanda"}
        page.evaluate(
            """(payload) => {
                const ta = document.querySelector('.body-param__text');
                if (ta) {
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(ta, JSON.stringify(payload, null, 2));
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }""", unseen)
        page.wait_for_timeout(500)
        page.click("button.execute")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SHOTS / "04_swagger_unseen_category.png"), full_page=True)
        print("  04_swagger_unseen_category.png")

        browser.close()

    print(f"Screenshots written to {SHOTS}")


if __name__ == "__main__":
    main()


def capture_console(log_text: str, title: str, out_name: str) -> Path:
    """Render real captured console output as a terminal-style image.

    Used because macOS blocks OS-level screencapture in this environment; the
    text is the verbatim output of the running server / demo client.
    """
    import html as _html
    doc = f"""<html><head><style>
      body {{ margin:0; background:#1e1e1e; font-family:'SF Mono',Menlo,monospace; }}
      .bar {{ background:#323233; color:#ccc; padding:8px 14px; font-size:13px;
              border-bottom:1px solid #000; }}
      .dot {{ height:12px; width:12px; border-radius:50%; display:inline-block;
              margin-right:6px; vertical-align:middle; }}
      pre {{ color:#d4d4d4; padding:16px; font-size:13px; line-height:1.45;
             white-space:pre-wrap; margin:0; }}
    </style></head><body>
      <div class="bar">
        <span class="dot" style="background:#ff5f56"></span>
        <span class="dot" style="background:#ffbd2e"></span>
        <span class="dot" style="background:#27c93f"></span>
        &nbsp;{_html.escape(title)}
      </div>
      <pre>{_html.escape(log_text)}</pre>
    </body></html>"""
    tmp = SHOTS / f"_{out_name}.html"
    tmp.write_text(doc)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(tmp.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(400)
        out = SHOTS / f"{out_name}.png"
        page.screenshot(path=str(out), full_page=True)
        browser.close()
    tmp.unlink(missing_ok=True)
    print(f"  {out.name}")
    return out
