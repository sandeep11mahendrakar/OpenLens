# ============================================================
# scrapers/reddit_scraper.py
# PURPOSE: Reddit content retrieval (best for Advice & Verify modes)
# ============================================================

import praw
import prawcore
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedditComment:
    """A single Reddit comment with metadata."""
    body: str
    score: int
    author: str
    created_utc: float
    is_top_level: bool
    awards_count: int
    word_count: int
    has_links: bool
    
    @property
    def quality_score(self) -> float:
        """
        Custom quality heuristic for ranking comments.
        
        Factors:
        - Upvote score (logarithmic — diminishing returns)
        - Length (longer ≈ more detailed, but not too long)
        - Awards (indicates community recognition)
        - Top-level (direct answers > nested replies)
        """
        import math
        
        score = 0.0
        
        # Upvote contribution (log scale to prevent domination)
        if self.score > 0:
            score += math.log(1 + self.score) * 2
        
        # Length bonus (sweet spot: 50-300 words)
        if 50 <= self.word_count <= 300:
            score += 3.0
        elif 30 <= self.word_count <= 500:
            score += 1.5
        elif self.word_count < 15:
            score -= 2.0  # Too short = probably not helpful
        
        # Awards bonus
        score += min(self.awards_count * 1.5, 5)  # Cap at 5
        
        # Top-level bonus
        if self.is_top_level:
            score += 2.0
        
        return score


@dataclass
class RedditThread:
    """A Reddit thread with its comments."""
    title: str
    selftext: str
    url: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: float
    comments: List[RedditComment]
    flair: Optional[str] = None
    
    def get_top_comments(self, n: int = 5) -> List[RedditComment]:
        """Get top N comments by quality score."""
        sorted_comments = sorted(
            self.comments,
            key=lambda c: c.quality_score,
            reverse=True
        )
        return sorted_comments[:n]


class RedditScraper:
    """
    Reddit content retrieval engine.
    
    WHY REDDIT?
    - Best source for advice/opinion content
    - Community-validated answers (upvote system)
    - Real human experiences and explanations
    - Great for fact-checking (r/IsItBullshit, r/AskScience, etc.)
    
    ARCHITECTURE:
    - Uses PRAW (official Reddit API wrapper)
    - Implements smart subreddit routing based on intent mode
    - Quality scoring algorithm for comment ranking
    """
    
    # Mode-to-subreddit mapping
    SUBREDDIT_MAP = {
        'advice': [
            'advice', 'NoStupidQuestions', 'AskReddit',
            'LifeProTips', 'personalfinance', 'careerguidance',
            'relationships', 'internetparents'
        ],
        'history': [
            'AskHistorians', 'history', 'HistoryMemes',
            'todayilearned', 'AskHistory'
        ],
        'science': [
            'askscience', 'science', 'explainlikeimfive',
            'EverythingScience', 'AskScienceDiscussion'
        ],
        'fact': [
            'NoStupidQuestions', 'explainlikeimfive',
            'todayilearned', 'AskReddit', 'answers'
        ],
        'verify': [
            'IsItBullshit', 'skeptic', 'DebunkThis',
            'askscience', 'NoStupidQuestions'
        ]
    }
    
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='OpenLens/1.0 Research Project'
        )
        # Set read-only mode (we're only scraping, not posting)
        self.reddit.read_only = True
    
    def search_subreddit(
        self, 
        query: str, 
        subreddit: str, 
        limit: int = 5,
        sort: str = 'relevance',
        time_filter: str = 'all'
    ) -> List[RedditThread]:
        """Search within a specific subreddit."""
        threads = []
        
        try:
            sub = self.reddit.subreddit(subreddit)
            results = sub.search(
                query, 
                sort=sort, 
                time_filter=time_filter, 
                limit=limit
            )
            
            for submission in results:
                # Skip removed/deleted posts
                if submission.removed_by_category or submission.selftext == '[deleted]':
                    continue
                
                # Get comments
                submission.comment_sort = 'best'
                submission.comments.replace_more(limit=0)  # Don't expand "more comments"
                
                comments = []
                for comment in submission.comments[:20]:  # Top 20 comments
                    if isinstance(comment, praw.models.MoreComments):
                        continue
                    if comment.body in ['[deleted]', '[removed]']:
                        continue
                    
                    comments.append(RedditComment(
                        body=self._clean_comment(comment.body),
                        score=comment.score,
                        author=str(comment.author) if comment.author else '[deleted]',
                        created_utc=comment.created_utc,
                        is_top_level=comment.parent_id.startswith('t3_'),
                        awards_count=getattr(comment, 'total_awards_received', 0),
                        word_count=len(comment.body.split()),
                        has_links=bool(re.search(r'https?://', comment.body))
                    ))
                
                threads.append(RedditThread(
                    title=submission.title,
                    selftext=self._clean_comment(submission.selftext),
                    url=f"https://reddit.com{submission.permalink}",
                    subreddit=subreddit,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    comments=comments,
                    flair=submission.link_flair_text
                ))
        
        except prawcore.exceptions.Redirect:
            print(f"Subreddit r/{subreddit} not found")
        except prawcore.exceptions.Forbidden:
            print(f"Access to r/{subreddit} is forbidden")
        except Exception as e:
            print(f"Error searching r/{subreddit}: {e}")
        
        return threads
    
    def smart_search(
        self, 
        query: str, 
        mode: str, 
        max_threads: int = 5
    ) -> List[RedditThread]:
        """
        Search across mode-appropriate subreddits.
        
        Strategy:
        1. Get target subreddits for this mode
        2. Search top 3 subreddits
        3. Merge and rank results
        4. Return top threads
        """
        target_subs = self.SUBREDDIT_MAP.get(mode, self.SUBREDDIT_MAP['fact'])
        
        all_threads = []
        
        # Search top 3 most relevant subreddits
        for subreddit in target_subs[:3]:
            threads = self.search_subreddit(
                query=query,
                subreddit=subreddit,
                limit=3,
                sort='relevance'
            )
            all_threads.extend(threads)
        
        # Rank threads by a combined score
        all_threads.sort(
            key=lambda t: self._thread_relevance_score(t, query),
            reverse=True
        )
        
        return all_threads[:max_threads]
    
    def _thread_relevance_score(self, thread: RedditThread, query: str) -> float:
        """Score a thread's relevance to the query."""
        import math
        
        score = 0.0
        query_words = set(query.lower().split())
        
        # Title relevance
        title_words = set(thread.title.lower().split())
        overlap = len(query_words & title_words) / max(len(query_words), 1)
        score += overlap * 10
        
        # Thread popularity (log scale)
        if thread.score > 0:
            score += math.log(1 + thread.score)
        
        # Comment count (more discussion = more content to extract)
        score += math.log(1 + thread.num_comments) * 0.5
        
        # Quality subreddit bonus
        quality_subs = {'AskHistorians', 'askscience', 'explainlikeimfive'}
        if thread.subreddit in quality_subs:
            score += 5  # These subs have strict quality standards
        
        # Has good comments
        if thread.comments:
            best_comment = max(thread.comments, key=lambda c: c.quality_score)
            score += best_comment.quality_score * 0.5
        
        return score
    
    def _clean_comment(self, text: str) -> str:
        """Clean Reddit comment text."""
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove Reddit-specific formatting
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        # Remove edit notices
        text = re.sub(r'Edit:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        return text.strip()


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == '__main__':
    scraper = RedditScraper()
    
    # Test advice mode
    threads = scraper.smart_search(
        "How to negotiate salary at a new job",
        mode='advice',
        max_threads=3
    )
    
    for thread in threads:
        print(f"\n{'='*60}")
        print(f"[r/{thread.subreddit}] {thread.title}")
        print(f"Score: {thread.score} | Comments: {thread.num_comments}")
        print(f"URL: {thread.url}")
        
        top_comments = thread.get_top_comments(3)
        for i, comment in enumerate(top_comments, 1):
            print(f"\n  Top Comment #{i} (score: {comment.score}, "
                  f"quality: {comment.quality_score:.1f}):")
            # Print first 200 chars
            preview = comment.body[:200] + "..." if len(comment.body) > 200 else comment.body
            print(f"  {preview}")