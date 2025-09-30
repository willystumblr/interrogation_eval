# google_claim_search.py
import json
import re
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pydantic import BaseModel, Field
import requests
from rank_bm25 import BM25Okapi

TOP_K_RESULTS = 3         # how many search results to fetch        # how many passages to return
_PAT = re.compile(r"(content|main|article|body|post)", re.I)
_SPLIT_RE = re.compile(r"\n{2,}")          # paragraph boundary = ≥2 new-lines
_TOKEN_RE = re.compile(r"\w+")

def _candidate_blocks(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """Return potential ‘main content’ elements in priority order."""
    # ① semantic tags
    blocks = soup.find_all(["main", "article"])
    if blocks:
        return blocks

    # ② id/class pattern match (limit scan to first ~40 nodes for speed)
    for tag in soup.find_all(["div", "section"], limit=40):
        ident = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
        if _PAT.search(ident):
            blocks.append(tag)
    return blocks


def _biggest_text_container(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Pick the best container for the article body.
    """
    total_len = len(soup.get_text(" ", strip=True))
    best, best_len = None, 0

    for cand in _candidate_blocks(soup):
        txt_len = len(cand.get_text(" ", strip=True))
        if txt_len > best_len:
            best, best_len = cand, txt_len

    # Fallback: largest <div>/<section> if patterns failed
    if not best:
        for cand in soup.find_all(["div", "section"], limit=30):
            txt_len = len(cand.get_text(" ", strip=True))
            if txt_len > best_len:
                best, best_len = cand, txt_len

    # sanity: only accept if it’s at least 25 % of whole-page text
    if best and best_len > 0.25 * total_len:
        return best
    return None


def _clean_html(html: str) -> str:
    """
    Extract the main article text as **Markdown** with *no* heavy dependencies.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 🔹 1. Drop obvious non-content nodes
    for tag in soup(["script", "style", "noscript", "header",
                     "footer", "nav", "aside", "iframe"]):
        tag.decompose()

    # 🔹 2. Choose the core container
    container = _biggest_text_container(soup) or soup.body or soup

    # 🔹 3. Convert to Markdown
    markdown = md(
        str(container),
        strip=["img", "iframe", "script", "style"],
        heading_style="ATX",
        bullets="*",
    )

    # 🔹 4. Whitespace tidy-up
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]{2,}", " ", markdown).strip()

    return markdown or "[content-extraction-failed]"


def _split_passages(text: str,
                    max_chars: int = 3000,
                    min_chars: int = 600) -> List[str]:
    """
    Split Markdown into ~paragraph-sized passages.

    1. First split on double-newline (natural paras).
    2. Merge consecutive short paras until >= min_chars.
    3. Hard-split any monster chunk longer than max_chars.
    """
    #print("text: ", text)
    
    parts, current, passages = _SPLIT_RE.split(text), [], []
    def flush():
        if current:
            chunk = "\n\n".join(current).strip()
            while len(chunk) > max_chars:   # hard split long tail
                passages.append(chunk[:max_chars])
                chunk = chunk[max_chars:]
            if chunk:
                passages.append(chunk)
            current.clear()

    for para in parts:
        if len(" ".join(current) + para) < min_chars:
            current.append(para)
        else:
            current.append(para)
            flush()
    flush()
    #print("passage: ", passages)
    #print()
    return passages

def _tokenize(txt: str) -> List[str]:
    return _TOKEN_RE.findall(txt.lower())


def _top_passages(claim: str,
                  passages: List[str],
                  k: int = 3) -> List[str]:
    if not passages:
        return []
    
    tokenized = [_tokenize(p) for p in passages]
    if all(len(toks) == 0 for toks in tokenized):
        return passages[:k]  # fallback

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(claim))
    ranked = sorted(zip(scores, passages), reverse=True)[:k]
    # keep only passages with non-zero score; fall back to first k otherwise
    picked = [p for s, p in ranked if s > 0] or passages[:k]
    #print('picked : ', picked)
    return picked

class GoogleClaimSearch(BaseModel):
    """
    LiteLLM-compatible tool:
    given a claim, return plain-text evidence passages from Google
    """
    api_key: str = Field(..., description="Google Custom Search API key")
    cx: str = Field(..., description="Google Programmable Search Engine ID")

    class Config:
        arbitrary_types_allowed = True

    def _fetch(self, url: str) -> Dict[str, Any]:
        try:
            page = requests.get(url, timeout=6,
                                headers={"User-Agent": "Mozilla/5.0"})
            page.raise_for_status()
            soup = BeautifulSoup(page.text, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""
            cleaned = _clean_html(page.text)
            return {"title": title, "cleaned": cleaned}
        except Exception as e:
            return {"title": "", "error": f"[Error fetching] {e}"}
    
    # ------------- tool entry point -------------
    def invoke(self, claim: str, q: str, gl: str) -> str:
        """
        Parameters
        ----------
        claim : str
            A factual claim / statement in natural language.
        
        q: str
            A keyword that needs to be searched for fact verification.
            This is usually a word/phrase within the claim.

        Returns
        -------
        str
            JSON stringified list[str] – each element is the cleaned text
            of a search-result page.  Errors become single-element lists.
        """
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            q_params = {
                "q": q,
                "key": self.api_key,
                "cx": self.cx,
                "num": TOP_K_RESULTS, # top 5 results
                "gl": gl,             # geolocation
            }
            resp = requests.get(search_url, params=q_params, timeout=6)
            resp.raise_for_status()
            items = resp.json().get("items", [])

            results: List[Dict[str, Any]] = []
            for it in items:
                url = it.get("link")
                if not url:
                    continue

                fetched = self._fetch(url)
                if "error" in fetched:
                    results.append({"query" : q, "title": fetched["title"], "link": url, "gl" : gl,
                                    "text_block": [fetched["error"]]})
                    continue
                passages = _split_passages(fetched["cleaned"])
                top_passages = _top_passages(claim, passages)
                results.append({
                    'query': q,
                    "title": fetched["title"],
                    "link": url,
                    "gl" : gl,
                    "text_block": top_passages})
                
                
                
            if not results:
                results = [{
                    "query": q,
                    "title": "",
                    "link": "",
                    "gl" : gl,
                    "text_block": ["No text could be extracted from the top results."]
                }]
            return json.dumps(results, ensure_ascii=False)

        except Exception as outer:
            return json.dumps([{"query" : q, "title": "", "link":"", "gl" : gl, "text_block":f"Search failure: {outer}"}], ensure_ascii=False)

    # ------------- schema that LiteLLM exports -------------
    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "google_claim_search",
                "description": (
                    "Given a factual `claim`, run Google Custom Search with the query (keyword) `q` and `gl`, "
                    f"crawl the top {TOP_K_RESULTS} result pages, and return a list of their plain texts "
                    "as a JSON string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "A claim or statement in natural language (provided by the user) that needs to be verified."
                        },
                        "q": {
                            "type": "string",
                            "description": "Query; a keyword that needs to be searched for fact verification."
                        },
                        "gl": {
                            "type": "string",
                            "description": "Geolocation of end user. The country code (e.g., 'us', 'uk', 'ca', 'jp', 'kr') to tailor search results to a specific region.",
                        },
                    },
                    "required": ["claim", "q", "gl"] #, "exactTerms", ],
                },
            },
        }