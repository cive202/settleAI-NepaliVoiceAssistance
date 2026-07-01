"""
rag/scraper.py — Recursive website scraping + chunking.

Responsibility:
  - Crawl a site starting at a URL, extracting plain text from each page
  - Split the resulting documents into embeddable chunks

Pages are rendered with a headless browser (Playwright) rather than fetched
as raw HTML, since plain requests can't see content or links injected by
client-side JavaScript (common on React/Vue/etc. sites).
"""

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright

from config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_MAX_PAGES


def _extract_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _extract_links(html: str, page_url: str) -> set[str]:
    base_domain = urlparse(page_url).netloc
    links = set()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = urljoin(page_url, a["href"]).split("#")[0]
        if (
            href.startswith(("http://", "https://"))
            and urlparse(href).netloc == base_domain
        ):
            links.add(href)
    return links


def scrape(url: str, max_depth: int) -> list[Document]:
    """Recursively crawl `url` up to `max_depth` links deep, staying on-domain."""
    visited: set[str] = set()
    frontier = {url}
    docs: list[Document] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            for depth in range(max_depth + 1):
                next_frontier: set[str] = set()
                for link in frontier - visited:
                    if len(visited) >= RAG_MAX_PAGES:
                        break
                    visited.add(link)
                    try:
                        page.goto(link, wait_until="networkidle", timeout=20000)
                    except Exception:
                        pass
                    try:
                        html = page.content()
                    except Exception:
                        continue
                    text = _extract_text(html)
                    if text:
                        docs.append(
                            Document(page_content=text, metadata={"source": link})
                        )
                    if depth < max_depth:
                        next_frontier |= _extract_links(html, link)
                frontier = next_frontier - visited
                if not frontier or len(visited) >= RAG_MAX_PAGES:
                    break
        finally:
            browser.close()

    return docs


def split(docs: list[Document]) -> list[Document]:
    # add_start_index lets retrieve_context() reassemble a page's chunks in
    # their original order when it pulls the full page back together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP, add_start_index=True
    )
    return splitter.split_documents(docs)
