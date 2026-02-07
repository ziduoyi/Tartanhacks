
import requests
from readability import Document
from bs4 import BeautifulSoup

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.text

def extract_main_text(html: str) -> str:
    doc = Document(html)
    soup = BeautifulSoup(doc.summary(), "html.parser")
    text = soup.get_text("\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text[:20000]
