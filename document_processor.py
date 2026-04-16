# ============================================================
# pipeline/document_processor.py
# PURPOSE: Chunk, embed, index, and retrieve documents
# THIS IS THE CORE RAG PIPELINE
# ============================================================

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class DocumentChunk:
    """
    A single chunk of text ready for embedding and retrieval.
    
    WHY CHUNKING MATTERS:
    - Embedding models have token limits (typically 512 tokens)
    - Smaller chunks = more precise retrieval
    - But too small = loss of context
    - Sweet spot: 150-300 words per chunk with overlap
    """
    chunk_id: str
    text: str
    source: str           # 'wikipedia' or 'reddit'
    source_url: str
    source_title: str
    section_title: str     # For Wikipedia sections
    word_count: int
    chunk_index: int       # Position in original document
    total_chunks: int      # Total chunks from this document
    metadata: Dict = field(default_factory=dict)


class TextChunker:
    """
    Intelligent text chunking with overlap.
    
    CHUNKING STRATEGIES (you should know these for interviews):
    
    1. Fixed-size: Simple but breaks mid-sentence
    2. Sentence-based: Respects sentence boundaries ← WE USE THIS
    3. Semantic: Uses embeddings to find topic shifts (advanced)
    4. Recursive: Tries multiple separators (LangChain approach)
    
    We use sentence-based with sliding window overlap.
    """
    
    def __init__(
        self, 
        target_chunk_size: int = 200,   # Target words per chunk
        overlap_size: int = 50,          # Words of overlap between chunks
        min_chunk_size: int = 50         # Minimum viable chunk
    ):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(
        self, 
        text: str, 
        source: str,
        source_url: str,
        source_title: str,
        section_title: str = "",
        metadata: Dict = None
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks at sentence boundaries."""
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_sentences = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > self.target_chunk_size and current_sentences:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_sentences)
                
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunk_id = hashlib.md5(
                        f"{source_title}:{section_title}:{len(chunks)}".encode()
                    ).hexdigest()[:12]
                    
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=source,
                        source_url=source_url,
                        source_title=source_title,
                        section_title=section_title,
                        word_count=len(chunk_text.split()),
                        chunk_index=len(chunks),
                        total_chunks=0,  # Updated after all chunks created
                        metadata=metadata or {}
                    ))
                
                # Keep overlap sentences for next chunk
                overlap_words = 0
                overlap_start = len(current_sentences)
                for i in range(len(current_sentences) - 1, -1, -1):
                    overlap_words += len(current_sentences[i].split())
                    if overlap_words >= self.overlap_size:
                        overlap_start = i
                        break
                
                current_sentences = current_sentences[overlap_start:]
                current_word_count = sum(len(s.split()) for s in current_sentences)
            
            current_sentences.append(sentence)
            current_word_count += sentence_words
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunk_id = hashlib.md5(
                    f"{source_title}:{section_title}:{len(chunks)}".encode()
                ).hexdigest()[:12]
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    source_url=source_url,
                    source_title=source_title,
                    section_title=section_title,
                    word_count=len(chunk_text.split()),
                    chunk_index=len(chunks),
                    total_chunks=0,
                    metadata=metadata or {}
                ))
        
        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


class EmbeddingEngine:
    """
    Generate embeddings for document chunks and queries.
    
    MODEL CHOICE: sentence-transformers/all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Only 80MB model size
    - Fast inference (CPU-friendly)
    - Good balance of quality vs speed
    - Top performer on MTEB benchmark for its size
    
    ALTERNATIVE for better quality: all-mpnet-base-v2 (768-dim, 420MB)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Returns numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (embedding_dim,)"""
        return self.model.encode(
            [query], 
            normalize_embeddings=True
        )[0]


class VectorIndex:
    """
    Simple but effective vector similarity search.
    
    For a project of this scale, we don't need Pinecone or Weaviate.
    A numpy-based index is perfectly fine for <100K chunks.
    
    For production scale, you'd swap this for FAISS or Qdrant.
    
    INTERVIEW TALKING POINT: "I built a custom vector index to understand
    the fundamentals, but I know when to use FAISS/Pinecone for scale."
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []
        self._is_built = False
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the index."""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.chunks.extend(chunks)
        self._is_built = True
        print(f"Index now contains {len(self.chunks)} chunks")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Find the most similar chunks to the query.
        
        Uses cosine similarity (since embeddings are L2-normalized,
        this is equivalent to dot product).
        """
        if not self._is_built:
            raise RuntimeError("Index is empty. Add chunks first.")
        
        # Cosine similarity = dot product for normalized vectors
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= score_threshold:
                results.append((self.chunks[idx], score))
        
        return results
    
    def search_with_diversity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        diversity_threshold: float = 0.8
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search with Maximum Marginal Relevance (MMR).
        
        WHY MMR: Prevents returning 5 chunks that all say the same thing.
        Balances relevance with diversity.
        
        This is what Perplexity and similar tools use internally.
        """
        if not self._is_built:
            raise RuntimeError("Index is empty. Add chunks first.")
        
        # Get all similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get candidates (top 3x what we need)
        candidate_count = min(top_k * 3, len(self.chunks))
        candidate_indices = np.argsort(similarities)[::-1][:candidate_count]
        
        # MMR selection
        selected = []
        remaining = list(candidate_indices)
        
        while len(selected) < top_k and remaining:
            if not selected:
                # First selection: highest similarity to query
                best_idx = remaining[0]
            else:
                # Subsequent: balance relevance vs diversity
                best_score = -1
                best_idx = remaining[0]
                
                for idx in remaining:
                    # Relevance to query
                    relevance = similarities[idx]
                    
                    # Similarity to already-selected chunks (we want LOW similarity)
                    selected_embeddings = self.embeddings[selected]
                    max_sim_to_selected = np.max(
                        np.dot(selected_embeddings, self.embeddings[idx])
                    )
                    
                    # MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
                    mmr_score = (diversity_threshold * relevance - 
                                (1 - diversity_threshold) * max_sim_to_selected)
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [
            (self.chunks[idx], float(similarities[idx])) 
            for idx in selected 
            if similarities[idx] >= 0.3
        ]
    
    def save(self, path: str = 'data/vector_index'):
        """Save index to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path / 'embeddings.npy', self.embeddings)
        
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'source': chunk.source,
                'source_url': chunk.source_url,
                'source_title': chunk.source_title,
                'section_title': chunk.section_title,
                'word_count': chunk.word_count,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'metadata': chunk.metadata,
            })
        
        with open(save_path / 'chunks.json', 'w') as f:
            json.dump(chunks_data, f)
        
        print(f"Saved index with {len(self.chunks)} chunks to {path}")
    
    @classmethod
    def load(cls, path: str = 'data/vector_index') -> 'VectorIndex':
        """Load index from disk."""
        load_path = Path(path)
        
        embeddings = np.load(load_path / 'embeddings.npy')
        
        with open(load_path / 'chunks.json', 'r') as f:
            chunks_data = json.load(f)
        
        chunks = [DocumentChunk(**cd) for cd in chunks_data]
        
        instance = cls(embedding_dim=embeddings.shape[1])
        instance.embeddings = embeddings
        instance.chunks = chunks
        instance._is_built = True
        
        print(f"Loaded index with {len(chunks)} chunks")
        return instance


# ============================================================
# FULL RETRIEVAL PIPELINE
# ============================================================
class RetrievalPipeline:
    """
    Orchestrates the full retrieval pipeline:
    Query → Scrape → Chunk → Embed → Index → Retrieve
    
    This is the class that ties everything together.
    """
    
    def __init__(self):
        from scrapers.wikipedia_scraper import WikipediaScraper
        from scrapers.reddit_scraper import RedditScraper
        
        self.wiki_scraper = WikipediaScraper(cache_enabled=True)
        self.reddit_scraper = RedditScraper()
        self.chunker = TextChunker(
            target_chunk_size=200,
            overlap_size=50,
            min_chunk_size=50
        )
        self.embedder = EmbeddingEngine()
        self.index = VectorIndex(embedding_dim=self.embedder.embedding_dim)
    
    def process_query(
        self, 
        query: str, 
        mode: str,
        top_k: int = 5,
        use_mmr: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Full pipeline: query → relevant chunks.
        
        Steps:
        1. Scrape relevant content based on mode
        2. Chunk the content
        3. Embed chunks
        4. Build temporary index
        5. Search with MMR
        """
        # Reset index for each query (in production, you'd have a persistent index)
        self.index = VectorIndex(embedding_dim=self.embedder.embedding_dim)
        
        all_chunks = []
        
        # Step 1: Scrape based on mode
        # Wikipedia is good for: history, science, fact, verify
        # Reddit is good for: advice, verify, fact
        
        if mode in ['history', 'science', 'fact']:
            wiki_articles = self.wiki_scraper.smart_search(query, mode, top_articles=3)
            for article in wiki_articles:
                # Chunk each section
                for section in article.sections:
                    chunks = self.chunker.chunk_text(
                        text=section.content,
                        source='wikipedia',
                        source_url=article.url,
                        source_title=article.title,
                        section_title=section.title,
                        metadata={
                            'references_count': article.references_count,
                            'categories': article.categories[:5]
                        }
                    )
                    all_chunks.extend(chunks)
                
                # Also chunk the summary
                if article.summary:
                    summary_chunks = self.chunker.chunk_text(
                        text=article.summary,
                        source='wikipedia',
                        source_url=article.url,
                        source_title=article.title,
                        section_title='Summary',
                        metadata={'is_summary': True}
                    )
                    all_chunks.extend(summary_chunks)
        
        if mode in ['advice', 'verify', 'fact']:
            reddit_threads = self.reddit_scraper.smart_search(query, mode, max_threads=5)
            for thread in reddit_threads:
                # Chunk top comments
                top_comments = thread.get_top_comments(5)
                for comment in top_comments:
                    chunks = self.chunker.chunk_text(
                        text=comment.body,
                        source='reddit',
                        source_url=thread.url,
                        source_title=thread.title,
                        section_title=f"r/{thread.subreddit}",
                        metadata={
                            'comment_score': comment.score,
                            'quality_score': comment.quality_score,
                            'subreddit': thread.subreddit
                        }
                    )
                    all_chunks.extend(chunks)
                
                # Also chunk the original post if it has content
                if thread.selftext and len(thread.selftext) > 50:
                    post_chunks = self.chunker.chunk_text(
                        text=thread.selftext,
                        source='reddit',
                        source_url=thread.url,
                        source_title=thread.title,
                        section_title=f"r/{thread.subreddit} (OP)",
                        metadata={'is_original_post': True}
                    )
                    all_chunks.extend(post_chunks)
        
        if not all_chunks:
            print("No content found for this query.")
            return []
        
        print(f"Created {len(all_chunks)} chunks from scraped content")
        
        # Step 2: Embed all chunks
        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = self.embedder.embed_texts(chunk_texts)
        
        # Step 3: Add to index
        self.index.add_chunks(all_chunks, chunk_embeddings)
        
        # Step 4: Embed query and search
        query_embedding = self.embedder.embed_query(query)
        
        if use_mmr:
            results = self.index.search_with_diversity(
                query_embedding, top_k=top_k
            )
        else:
            results = self.index.search(
                query_embedding, top_k=top_k
            )
        
        return results