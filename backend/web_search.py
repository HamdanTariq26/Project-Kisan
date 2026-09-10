from urllib.parse import urlparse
import urllib.request

from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .rag import reranker



def retrieve_web_documents(query: str, target_sites: int = 5, search_pool: int = 15):
    """
    Search broadly and keep collecting usable webpages until
    target_sites are found or search results are exhausted.
    """

    try:
        with DDGS() as ddgs:
            search_results = list(
                ddgs.text(
                    query,
                    max_results=search_pool,
                )
            )

    except Exception as e:
        print(f"Web search failed: {e}")
        return []

    if not search_results:
        return []

    websites = []
    seen_urls = set()
    seen_domains = set()

    for result in search_results:
        if len(websites) >= target_sites:
            break

        title = result.get("title", "Unknown")
        url = result.get("href")

        if not url:
            continue

        normalized_url = url.rstrip("/").lower()

        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)

        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            continue

        if domain in seen_domains:
            continue

        text = fetch_webpage(url)

        if not text:
            continue

        seen_domains.add(domain)

        websites.append({
            "title": title,
            "url": url,
            "domain": domain,
            "content": text,
        })

    return websites



def fetch_webpage(url: str, timeout: int = 10):
    """
    Download a webpage and extract readable text.

    Returns:
        str | None
    """

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/131.0 Safari/537.36"
                )
            },
        )

        response = urllib.request.urlopen(req, timeout=timeout)
        html = response.read()

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "form",
            "aside",
        ]):
            tag.decompose()

        elements = soup.find_all([
            "h1",
            "h2",
            "h3",
            "p",
            "li",
        ])

        texts = []

        for element in elements:
            text = element.get_text(" ", strip=True)

            if text:
                texts.append(text)

        text_content = "\n".join(texts).strip()

        if not text_content:
            return None

        return text_content

    except Exception:
        return None


web_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def chunk_web_documents(websites):
    chunks = []

    for website in websites:
        split_text = web_text_splitter.split_text(website["content"])

        for chunk in split_text:
            chunks.append({
                "title": website["title"],
                "url": website["url"],
                "domain": website["domain"],
                "content": chunk,
            })

    return chunks



def rerank_web_documents(
    query: str,
    chunks,
    final_k: int = 10,
    max_per_site: int = 3,
    min_score: float = -999,
):
    if not chunks:
        return []

    pairs = [
        [query, chunk["content"]]
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    site_counts = {}

    for chunk, score in ranked:
        if score < min_score:
            continue

        domain = chunk["domain"]
        count = site_counts.get(domain, 0)

        if count >= max_per_site:
            continue

        result = chunk.copy()
        result["rerank_score"] = float(score)

        results.append(result)

        site_counts[domain] = count + 1

        if len(results) >= final_k:
            break

    return results



def web_search(query: str):
    websites = retrieve_web_documents(
        query=query,
        target_sites=5,
        search_pool=15,
    )

    if not websites:
        return {
            "status": "web_search_failed",
            "results": [],
        }

    chunks = chunk_web_documents(websites)

    if not chunks:
        return {
            "status": "web_search_failed",
            "results": [],
        }

    ranked_chunks = rerank_web_documents(
        query=query,
        chunks=chunks,
        final_k=10,
        max_per_site=3,
    )

    if not ranked_chunks:
        return {
            "status": "web_search_failed",
            "results": [],
        }

    return {
        "status": "web_results_found",
        "results": [
            {
                "title": chunk["title"],
                "url": chunk["url"],
                "content": chunk["content"],
                "rerank_score": chunk["rerank_score"],
            }
            for chunk in ranked_chunks
        ],
    }



def web_search_tool(query):
    return web_search(query)
