# ============================================================
# scrapers/wikipedia_scraper.py
# PURPOSE: Intelligent Wikipedia content retrieval
# ============================================================

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import quote
import re
import time
import hashlib
import json
from pathlib import Path


@dataclass
class WikiSection:
    """A parsed section from a Wikipedia article."""
    title: str
    level: int              # h2 = 2, h3 = 3, etc.
    content: str            # Raw text content
    sentences: List[str]    # Individual sentences
    word_count: int
    has_citations: bool
    parent_section: Optional[str] = None


@dataclass
class WikiArticle:
    """Complete parsed Wikipedia article."""
    title: str
    url: str
    summary: str
    sections: List[WikiSection]
    categories: List[str]
    references_count: int
    total_word_count: int
    infobox: Dict[str, str] = field(default_factory=dict)
    disambiguation: bool = False
    
    def get_relevant_sections(self, keywords: List[str], top_k: int = 3) -> List[WikiSection]:
        """
        Rank sections by keyword relevance.
        This is a simple BM25-like scoring before we add embeddings in Month 2.
        """
        scored_sections = []
        for section in self.sections:
            score = 0
            content_lower = section.content.lower()
            for keyword in keywords:
                # Term frequency
                tf = content_lower.count(keyword.lower())
                # Boost for title matches
                if keyword.lower() in section.title.lower():
                    tf += 5
                score += tf
            
            # Normalize by document length (prevents bias toward long sections)
            if section.word_count > 0:
                score = score / (1 + 0.5 * (section.word_count / 200))
            
            scored_sections.append((section, score))
        
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        return [s for s, score in scored_sections[:top_k] if score > 0]


class WikipediaScraper:
    """
    Production-grade Wikipedia scraper.
    
    DESIGN DECISIONS:
    1. Uses Wikipedia API first (faster, more reliable)
    2. Falls back to HTML scraping for structured data (tables, infoboxes)
    3. Implements caching to avoid redundant requests
    4. Respects rate limits and robots.txt
    5. Handles disambiguation pages gracefully
    """
    
    API_BASE = "https://en.wikipedia.org/w/api.php"
    WEB_BASE = "https://en.wikipedia.org/wiki/"
    CACHE_DIR = Path("cache/wikipedia")
    
    # Sections to skip (they don't contain useful answer content)
    SKIP_SECTIONS = {
        'references', 'external links', 'see also', 'further reading',
        'notes', 'bibliography', 'sources', 'citations'
    }
    
    def __init__(self, cache_enabled: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenLens/1.0 (Research Project; Contact: your@email.com)'
        })
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._request_count = 0
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting — be a good citizen."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 0.1:  # Max 10 requests per second
            time.sleep(0.1 - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _check_cache(self, query: str) -> Optional[Dict]:
        """Check if we have a cached result."""
        if not self.cache_enabled:
            return None
        cache_file = self.CACHE_DIR / f"{self._get_cache_key(query)}.json"
        if cache_file.exists():
            # Cache expires after 24 hours
            if time.time() - cache_file.stat().st_mtime < 86400:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        return None
    
    def _save_cache(self, query: str, data: Dict):
        """Save result to cache."""
        if not self.cache_enabled:
            return
        cache_file = self.CACHE_DIR / f"{self._get_cache_key(query)}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search Wikipedia for relevant articles.
        
        Returns list of {title, snippet, pageid} dicts.
        """
        cached = self._check_cache(f"search:{query}")
        if cached:
            return cached
        
        self._rate_limit()
        
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': num_results,
            'srinfo': 'totalhits|suggestion',
            'srprop': 'snippet|titlesnippet|wordcount',
            'format': 'json'
        }
        
        response = self.session.get(self.API_BASE, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Clean HTML from snippets
            clean_snippet = BeautifulSoup(item['snippet'], 'html.parser').get_text()
            results.append({
                'title': item['title'],
                'snippet': clean_snippet,
                'pageid': item['pageid'],
                'wordcount': item.get('wordcount', 0)
            })
        
        self._save_cache(f"search:{query}", results)
        return results
    
    def get_article(self, title: str) -> WikiArticle:
        """
        Fetch and parse a complete Wikipedia article.
        
        This is the heavy lifting — we parse the HTML to extract
        structured sections, infoboxes, and clean text.
        """
        cached = self._check_cache(f"article:{title}")
        if cached:
            return self._dict_to_article(cached)
        
        # Step 1: Get article summary via API
        summary = self._get_summary(title)
        
        # Step 2: Get full HTML for structured parsing
        html_content = self._get_html(title)
        
        # Step 3: Parse into structured sections
        sections = self._parse_sections(html_content)
        
        # Step 4: Extract infobox if present
        infobox = self._parse_infobox(html_content)
        
        # Step 5: Get categories
        categories = self._get_categories(title)
        
        article = WikiArticle(
            title=title,
            url=f"{self.WEB_BASE}{quote(title.replace(' ', '_'))}",
            summary=summary,
            sections=sections,
            categories=categories,
            references_count=self._count_references(html_content),
            total_word_count=sum(s.word_count for s in sections),
            infobox=infobox,
            disambiguation=self._is_disambiguation(html_content)
        )
        
        # Cache the result
        self._save_cache(f"article:{title}", self._article_to_dict(article))
        
        return article
    
    def _get_summary(self, title: str) -> str:
        """Get article summary via REST API."""
        self._rate_limit()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json().get('extract', '')
        return ''
    
    def _get_html(self, title: str) -> BeautifulSoup:
        """Get parsed HTML of the article."""
        self._rate_limit()
        params = {
            'action': 'parse',
            'page': title,
            'prop': 'text',
            'format': 'json',
            'disabletoc': True
        }
        response = self.session.get(self.API_BASE, params=params)
        response.raise_for_status()
        html = response.json()['parse']['text']['*']
        return BeautifulSoup(html, 'html.parser')
    
    def _parse_sections(self, soup: BeautifulSoup) -> List[WikiSection]:
        """
        Parse HTML into structured sections.
        
        This handles the messy reality of Wikipedia HTML:
        - Nested sections (h2 > h3 > h4)
        - Tables mixed with text
        - Citation references [1][2]
        - Hidden elements
        """
        sections = []
        current_section = None
        current_content = []
        
        # Remove unwanted elements
        for element in soup.find_all(['sup', 'style', 'script']):
            element.decompose()
        
        for element in soup.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol']):
            if element.name in ['h2', 'h3', 'h4']:
                # Save previous section
                if current_section and current_content:
                    text = ' '.join(current_content)
                    # Split into sentences
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                    
                    if current_section.lower() not in self.SKIP_SECTIONS:
                        sections.append(WikiSection(
                            title=current_section,
                            level=int(element.name[1]),
                            content=text,
                            sentences=sentences,
                            word_count=len(text.split()),
                            has_citations=bool(re.search(r'\[\d+\]', text))
                        ))
                
                # Start new section
                headline = element.find('span', class_='mw-headline')
                current_section = headline.get_text() if headline else element.get_text()
                current_content = []
            
            elif element.name == 'p':
                text = element.get_text().strip()
                # Clean citation markers
                text = re.sub(r'\[\d+\]', '', text)
                text = re.sub(r'\s+', ' ', text)
                if text and len(text) > 20:  # Skip tiny paragraphs
                    current_content.append(text)
            
            elif element.name in ['ul', 'ol']:
                items = element.find_all('li')
                for item in items[:10]:  # Cap at 10 items
                    item_text = item.get_text().strip()
                    item_text = re.sub(r'\[\d+\]', '', item_text)
                    if item_text and len(item_text) > 10:
                        current_content.append(f"• {item_text}")
        
        # Don't forget the last section
        if current_section and current_content:
            text = ' '.join(current_content)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if current_section.lower() not in self.SKIP_SECTIONS:
                sections.append(WikiSection(
                    title=current_section,
                    level=2,
                    content=text,
                    sentences=sentences,
                    word_count=len(text.split()),
                    has_citations=bool(re.search(r'\[\d+\]', text))
                ))
        
        return sections
    
    def _parse_infobox(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract infobox data (the sidebar table on Wikipedia articles)."""
        infobox = {}
        table = soup.find('table', class_='infobox')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                header = row.find('th')
                data = row.find('td')
                if header and data:
                    key = header.get_text().strip()
                    value = data.get_text().strip()
                    # Clean up
                    value = re.sub(r'\[\d+\]', '', value)
                    value = re.sub(r'\s+', ' ', value)
                    if key and value:
                        infobox[key] = value
        return infobox
    
    def _get_categories(self, title: str) -> List[str]:
        """Get article categories."""
        self._rate_limit()
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'categories',
            'cllimit': 20,
            'clshow': '!hidden',
            'format': 'json'
        }
        response = self.session.get(self.API_BASE, params=params)
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        
        categories = []
        for page in pages.values():
            for cat in page.get('categories', []):
                cat_name = cat['title'].replace('Category:', '')
                categories.append(cat_name)
        return categories
    
    def _count_references(self, soup: BeautifulSoup) -> int:
        """Count the number of references (indicator of article quality)."""
        ref_list = soup.find('ol', class_='references')
        if ref_list:
            return len(ref_list.find_all('li'))
        return 0
    
    def _is_disambiguation(self, soup: BeautifulSoup) -> bool:
        """Check if this is a disambiguation page."""
        return bool(soup.find('table', id='disambig') or 
                    soup.find('div', class_='dmbox'))
    
    def _article_to_dict(self, article: WikiArticle) -> Dict:
        """Convert article to dict for caching."""
        return {
            'title': article.title,
            'url': article.url,
            'summary': article.summary,
            'sections': [
                {
                    'title': s.title,
                    'level': s.level,
                    'content': s.content,
                    'sentences': s.sentences,
                    'word_count': s.word_count,
                    'has_citations': s.has_citations,
                }
                for s in article.sections
            ],
            'categories': article.categories,
            'references_count': article.references_count,
            'total_word_count': article.total_word_count,
            'infobox': article.infobox,
            'disambiguation': article.disambiguation,
        }
    
    def _dict_to_article(self, d: Dict) -> WikiArticle:
        """Convert dict back to WikiArticle."""
        sections = [
            WikiSection(**s) for s in d['sections']
        ]
        return WikiArticle(
            title=d['title'],
            url=d['url'],
            summary=d['summary'],
            sections=sections,
            categories=d['categories'],
            references_count=d['references_count'],
            total_word_count=d['total_word_count'],
            infobox=d.get('infobox', {}),
            disambiguation=d.get('disambiguation', False),
        )
    
    def smart_search(self, query: str, mode: str, top_articles: int = 3) -> List[WikiArticle]:
        """
        High-level search that adapts strategy based on intent mode.
        
        MODE-SPECIFIC STRATEGIES:
        - HISTORY: Prioritize articles with timeline sections, "Background" sections
        - SCIENCE: Look for articles with "Mechanism", "Process" sections
        - FACT: Prioritize infobox data and article summaries
        - VERIFY: Search for the claim + "myth" or "misconception"
        - ADVICE: Not great for Wikipedia — this will mostly use Reddit
        """
        search_queries = self._expand_query(query, mode)
        
        all_articles = []
        seen_titles = set()
        
        for search_query in search_queries:
            results = self.search(search_query, num_results=3)
            for result in results:
                if result['title'] not in seen_titles:
                    seen_titles.add(result['title'])
                    article = self.get_article(result['title'])
                    if not article.disambiguation:
                        all_articles.append(article)
                
                if len(all_articles) >= top_articles:
                    break
            
            if len(all_articles) >= top_articles:
                break
        
        return all_articles
    
    def _expand_query(self, query: str, mode: str) -> List[str]:
        """
        Expand user query into multiple search queries.
        
        This is query expansion — a standard IR technique.
        """
        queries = [query]  # Always include original
        
        if mode == 'history':
            queries.append(f"{query} history timeline")
            queries.append(f"{query} historical event")
        elif mode == 'science':
            queries.append(f"{query} scientific explanation")
            queries.append(f"{query} mechanism process")
        elif mode == 'verify':
            queries.append(f"{query} myth misconception")
            queries.append(f"{query} scientific evidence")
        elif mode == 'fact':
            # For facts, the original query is usually specific enough
            pass
        elif mode == 'advice':
            queries.append(f"{query} guide overview")
        
        return queries[:3]  # Max 3 search queries


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == '__main__':
    scraper = WikipediaScraper(cache_enabled=True)
    
    # Example: History mode query
    articles = scraper.smart_search(
        "What caused the fall of the Roman Empire?",
        mode='history',
        top_articles=2
    )
    
    for article in articles:
        print(f"\n{'='*60}")
        print(f"Title: {article.title}")
        print(f"URL: {article.url}")
        print(f"Word count: {article.total_word_count}")
        print(f"References: {article.references_count}")
        print(f"Sections: {len(article.sections)}")
        
        if article.infobox:
            print(f"\nInfobox:")
            for k, v in list(article.infobox.items())[:5]:
                print(f"  {k}: {v}")
        
        # Get relevant sections
        relevant = article.get_relevant_sections(
            ['fall', 'decline', 'cause', 'collapse'],
            top_k=3
        )
        print(f"\nMost relevant sections:")
        for section in relevant:
            print(f"  - {section.title} ({section.word_count} words)")
            # Print first 2 sentences
            for sent in section.sentences[:2]:
                print(f"    {sent}")