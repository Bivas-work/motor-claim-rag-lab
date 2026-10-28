"""Explore one FCA Handbook page to find the element that holds the rule text.

Run from the project root with the .venv Python:
    uv run python ingest/explore.py
"""
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_FILE = PROJECT_ROOT / "data" / "raw" / "html" / "ICOBS_8_1.html"

# Replace with the selector you found in the browser's Inspect panel,
# e.g. "div.some-class" or "#some-id".
SELECTOR = "div.provision"

html = HTML_FILE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "lxml")

print("Page title:", soup.title.get_text(strip=True) if soup.title else None)

content = soup.select_one(SELECTOR)
if content is None:
    print(f"No element matched {SELECTOR!r}. Check the selector and try again.")
else:
    text = content.get_text("\n", strip=True)
    print(f"Matched {len(text)} characters. First 2000:\n")
    print(text[:2000])
