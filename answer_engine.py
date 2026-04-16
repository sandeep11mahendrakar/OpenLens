# ============================================================
# synthesis/answer_engine.py
# PURPOSE: Synthesize retrieved chunks into a coherent answer
# THIS IS WHAT MAKES IT FEEL LIKE AN "LLM RESPONSE"
# ============================================================

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import re
from collections import Counter


@dataclass
class SourceCitation:
    """A citation reference for the answer."""
    index: int
    title: str
    url: str
    source_type: str   # 'wikipedia' or 'reddit'
    relevance_score: float


@dataclass
class SynthesizedAnswer:
    """The final formatted answer."""
    query: str
    mode: str
    answer_text: str
    confidence: float
    citations: List[SourceCitation]
    key_points: List[str]
    metadata: Dict
    

class AnswerSynthesizer:
    """
    Synthesizes retrieved chunks into a coherent, formatted answer.
    
    THIS IS THE HARD PART and where you show real engineering skill.
    
    Without an LLM to generate text, we use:
    1. Extractive summarization (TextRank algorithm)
    2. Sentence ordering and deduplication
    3. Mode-specific formatting templates
    4. Citation integration
    
    INTERVIEW TALKING POINT: "I built this without relying on an LLM API,
    which forced me to understand extractive NLP techniques deeply."
    """
    
    # Mode-specific response templates
    TEMPLATES = {
        'advice': {
            'intro': "Based on community discussions and expert recommendations:\n\n",
            'section_headers': True,
            'use_bullet_points': True,
            'include_caveats': True,
            'tone': 'helpful'
        },
        'history': {
            'intro': "Here's what historical sources indicate:\n\n",
            'section_headers': True,
            'use_bullet_points': False,
            'include_caveats': False,
            'tone': 'informative',
            'chronological': True
        },
        'science': {
            'intro': "Here's the scientific explanation:\n\n",
            'section_headers': True,
            'use_bullet_points': False,
            'include_caveats': True,
            'tone': 'educational'
        },
        'fact': {
            'intro': "",  # Facts should be direct
            'section_headers': False,
            'use_bullet_points': False,
            'include_caveats': False,
            'tone': 'concise'
        },
        'verify': {
            'intro': "After examining available sources:\n\n",
            'section_headers': True,
            'use_bullet_points': True,
            'include_caveats': True,
            'tone': 'analytical',
            'verdict_required': True
        }
    }
    
    def __init__(self):
        self._setup_nlp()
    
    def _setup_nlp(self):
        """Initialize NLP tools for sentence processing."""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def synthesize(
        self,
        query: str,
        mode: str,
        retrieved_chunks: List[Tuple],  # List of (DocumentChunk, score)
        max_sentences: int = 12
    ) -> SynthesizedAnswer:
        """
        Main synthesis pipeline.
        
        Steps:
        1. Extract all sentences from retrieved chunks
        2. Remove duplicate/near-duplicate sentences
        3. Rank sentences using TextRank
        4. Order sentences logically
        5. Format according to mode template
        6. Add citations
        """
        if not retrieved_chunks:
            return SynthesizedAnswer(
                query=query,
                mode=mode,
                answer_text="I couldn't find enough information to answer this question. "
                           "Try rephrasing your query or asking something more specific.",
                confidence=0.0,
                citations=[],
                key_points=[],
                metadata={'status': 'no_results'}
            )
        
        # Step 1: Extract and score all sentences
        all_sentences = self._extract_sentences(retrieved_chunks)
        
        # Step 2: Remove duplicates
        unique_sentences = self._deduplicate_sentences(all_sentences)
        
        # Step 3: Rank using TextRank
        ranked_sentences = self._textrank_summarize(
            unique_sentences, 
            query,
            top_n=max_sentences
        )
        
        # Step 4: Extract key points
        key_points = self._extract_key_points(ranked_sentences, query)
        
        # Step 5: Format the answer
        template = self.TEMPLATES.get(mode, self.TEMPLATES['fact'])
        formatted_answer = self._format_answer(
            ranked_sentences, 
            template, 
            mode,
            query
        )
        
        # Step 6: Build citations
        citations = self._build_citations(retrieved_chunks)
        
        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(retrieved_chunks, unique_sentences)
        
        # Step 8: Add citations to answer text
        answer_with_citations = self._integrate_citations(
            formatted_answer, citations
        )
        
        return SynthesizedAnswer(
            query=query,
            mode=mode,
            answer_text=answer_with_citations,
            confidence=confidence,
            citations=citations,
            key_points=key_points,
            metadata={
                'chunks_used': len(retrieved_chunks),
                'sentences_extracted': len(all_sentences),
                'unique_sentences': len(unique_sentences),
                'sources': {
                    'wikipedia': sum(1 for c, _ in retrieved_chunks if c.source == 'wikipedia'),
                    'reddit': sum(1 for c, _ in retrieved_chunks if c.source == 'reddit')
                }
            }
        )
    
    def _extract_sentences(
        self, 
        chunks: List[Tuple]
    ) -> List[Dict]:
        """Extract individual sentences with metadata."""
        from nltk.tokenize import sent_tokenize
        
        sentences = []
        for chunk, relevance_score in chunks:
            sents = sent_tokenize(chunk.text)
            for i, sent in enumerate(sents):
                sent = sent.strip()
                if len(sent.split()) < 5:  # Skip very short sentences
                    continue
                if len(sent.split()) > 60:  # Skip overly long ones
                    continue
                    
                sentences.append({
                    'text': sent,
                    'source': chunk.source,
                    'source_title': chunk.source_title,
                    'source_url': chunk.source_url,
                    'relevance_score': relevance_score,
                    'position': i / max(len(sents), 1),  # Normalized position
                    'chunk_id': chunk.chunk_id
                })
        
        return sentences
    
    def _deduplicate_sentences(
        self, 
        sentences: List[Dict], 
        similarity_threshold: float = 0.75
    ) -> List[Dict]:
        """
        Remove near-duplicate sentences.
        Uses Jaccard similarity on word sets for efficiency.
        """
        unique = []
        
        for sent in sentences:
            words = set(sent['text'].lower().split())
            is_duplicate = False
            
            for existing in unique:
                existing_words = set(existing['text'].lower().split())
                
                # Jaccard similarity
                if words and existing_words:
                    intersection = len(words & existing_words)
                    union = len(words | existing_words)
                    similarity = intersection / union
                    
                    if similarity > similarity_threshold:
                        # Keep the one with higher relevance score
                        if sent['relevance_score'] > existing['relevance_score']:
                            unique.remove(existing)
                            unique.append(sent)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(sent)
        
        return unique
    
    def _textrank_summarize(
        self,
        sentences: List[Dict],
        query: str,
        top_n: int = 10,
        query_boost: float = 0.3
    ) -> List[Dict]:
        """
        TextRank-based extractive summarization.
        
        This is the same algorithm Google used in early search!
        
        Modifications from standard TextRank:
        - Query-biased scoring (boost sentences relevant to the query)
        - Position bias (early sentences in a document are more important)
        - Source quality weighting
        """
        from nltk.corpus import stopwords
        
        if len(sentences) <= top_n:
            return sentences
        
        stop_words = set(stopwords.words('english'))
        
        # Tokenize sentences into word sets (excluding stopwords)
        def get_words(text):
            return {
                w.lower() for w in re.findall(r'\b\w+\b', text) 
                if w.lower() not in stop_words and len(w) > 2
            }
        
        sentence_words = [get_words(s['text']) for s in sentences]
        query_words = get_words(query)
        
        n = len(sentences)
        
        # Build similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if sentence_words[i] and sentence_words[j]:
                    # Jaccard-like similarity
                    intersection = len(sentence_words[i] & sentence_words[j])
                    union = len(sentence_words[i] | sentence_words[j])
                    sim = intersection / union if union > 0 else 0
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        # Power iteration for PageRank-style scoring
        scores = np.ones(n) / n
        damping = 0.85
        
        for _ in range(30):  # 30 iterations is usually enough
            new_scores = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and similarity_matrix[j].sum() > 0:
                        new_scores[i] += (
                            similarity_matrix[j][i] / similarity_matrix[j].sum() * scores[j]
                        )
                new_scores[i] = (1 - damping) / n + damping * new_scores[i]
            scores = new_scores
        
        # Apply query bias
        for i in range(n):
            if sentence_words[i] and query_words:
                query_overlap = len(sentence_words[i] & query_words) / len(query_words)
                scores[i] += query_boost * query_overlap
            
            # Position bias (sentences earlier in their document get a small boost)
            scores[i] += 0.1 * (1 - sentences[i]['position'])
            
            # Source quality bias
            scores[i] *= sentences[i]['relevance_score']
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Sort by original order for readability
        top_indices = sorted(top_indices)
        
        return [sentences[i] for i in top_indices]
    
    def _extract_key_points(
        self, 
        sentences: List[Dict],
        query: str,
        max_points: int = 5
    ) -> List[str]:
        """Extract bullet-point-worthy key takeaways."""
        key_points = []
        
        # Look for sentences with strong signal words
        signal_patterns = [
            r'\b(main|key|important|significant|primary|major)\b',
            r'\b(because|therefore|consequently|as a result)\b',
            r'\b(first|second|third|finally|additionally)\b',
            r'\b(studies show|research indicates|evidence suggests)\b',
            r'\b(in conclusion|overall|in summary)\b',
        ]
        
        for sent in sentences:
            text = sent['text']
            for pattern in signal_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Clean and truncate
                    point = text.strip()
                    if len(point) > 150:
                        # Cut at last sentence boundary before 150 chars
                        point = point[:150].rsplit('.', 1)[0] + '.'
                    key_points.append(point)
                    break
            
            if len(key_points) >= max_points:
                break
        
        return key_points
    
    def _format_answer(
        self,
        sentences: List[Dict],
        template: Dict,
        mode: str,
        query: str
    ) -> str:
        """Format sentences into a coherent answer based on mode template."""
        parts = []
        
        # Add intro
        if template.get('intro'):
            parts.append(template['intro'])
        
        if template.get('use_bullet_points'):
            # Bullet point format (advice, verify)
            for sent in sentences:
                parts.append(f"• {sent['text']}")
            answer = '\n'.join(parts)
        
        elif template.get('chronological'):
            # Paragraph format with chronological feel (history)
            paragraphs = self._group_into_paragraphs(sentences, max_per_paragraph=3)
            for para in paragraphs:
                para_text = ' '.join(s['text'] for s in para)
                parts.append(para_text)
            answer = '\n\n'.join(parts)
        
        else:
            # Standard paragraph format (science, fact)
            paragraphs = self._group_into_paragraphs(sentences, max_per_paragraph=4)
            for para in paragraphs:
                para_text = ' '.join(s['text'] for s in para)
                parts.append(para_text)
            answer = '\n\n'.join(parts)
        
        # Add verdict for verify mode
        if mode == 'verify':
            verdict = self._generate_verdict(sentences)
            answer = f"**Verdict: {verdict}**\n\n{answer}"
        
        # Add caveats if needed
        if template.get('include_caveats'):
            answer += "\n\n*Note: This answer is synthesized from publicly available sources. " \
                     "For critical decisions, please consult authoritative sources or experts.*"
        
        return answer
    
    def _group_into_paragraphs(
        self, 
        sentences: List[Dict], 
        max_per_paragraph: int = 3
    ) -> List[List[Dict]]:
        """Group sentences into logical paragraphs."""
        paragraphs = []
        current_para = []
        current_source = None
        
        for sent in sentences:
            # Start new paragraph if source changes or we hit max
            if (len(current_para) >= max_per_paragraph or 
                (current_source and sent['source_title'] != current_source)):
                if current_para:
                    paragraphs.append(current_para)
                current_para = []
            
            current_para.append(sent)
            current_source = sent['source_title']
        
        if current_para:
            paragraphs.append(current_para)
        
        return paragraphs
    
    def _generate_verdict(self, sentences: List[Dict]) -> str:
        """
        For VERIFY mode, generate a verdict based on the evidence.
        
        This is a simple heuristic approach:
        - Look for confirmation/denial language in the sources
        - Weight by source quality
        """
        confirmation_words = {'true', 'correct', 'accurate', 'confirmed', 'proven',
                             'indeed', 'actually', 'verified', 'supported', 'evidence'}
        denial_words = {'false', 'incorrect', 'myth', 'debunked', 'misconception',
                       'untrue', 'no evidence', 'disproven', 'misleading', 'wrong'}
        
        confirm_score = 0
        deny_score = 0
        
        for sent in sentences:
            words = set(sent['text'].lower().split())
            confirm_hits = len(words & confirmation_words)
            deny_hits = len(words & denial_words)
            
            weight = sent['relevance_score']
            confirm_score += confirm_hits * weight
            deny_score += deny_hits * weight
        
        if deny_score > confirm_score * 1.5:
            return "Mostly FALSE / Misleading ❌"
        elif confirm_score > deny_score * 1.5:
            return "Mostly TRUE ✅"
        elif confirm_score > 0 or deny_score > 0:
            return "MIXED / Partially True ⚠️"
        else:
            return "INCONCLUSIVE — Not enough evidence to determine 🤔"
    
    def _build_citations(self, chunks: List[Tuple]) -> List[SourceCitation]:
        """Build citation list from used chunks."""
        seen_urls = set()
        citations = []
        
        for chunk, score in chunks:
            if chunk.source_url not in seen_urls:
                seen_urls.add(chunk.source_url)
                citations.append(SourceCitation(
                    index=len(citations) + 1,
                    title=chunk.source_title,
                    url=chunk.source_url,
                    source_type=chunk.source,
                    relevance_score=score
                ))
        
        return citations
    
    def _integrate_citations(
        self, 
        answer: str, 
        citations: List[SourceCitation]
    ) -> str:
        """Add citation references to the answer."""
        if not citations:
            return answer
        
        # Add sources section at the bottom
        sources_section = "\n\n---\n📚 **Sources:**\n"
        for cite in citations:
            icon = "📖" if cite.source_type == 'wikipedia' else "💬"
            sources_section += f"  [{cite.index}] {icon} {cite.title}\n"
            sources_section += f"      {cite.url}\n"
        
        return answer + sources_section
    
    def _calculate_confidence(
        self, 
        chunks: List[Tuple],
        sentences: List[Dict]
    ) -> float:
        """
        Calculate overall confidence in the answer.
        
        Factors:
        - Average relevance of retrieved chunks
        - Number of sources (more = more confident)
        - Source diversity (both Wikipedia + Reddit = better)
        - Number of sentences extracted
        """
        if not chunks:
            return 0.0
        
        # Average retrieval score
        avg_relevance = sum(score for _, score in chunks) / len(chunks)
        
        # Source count factor
        unique_sources = len(set(c.source_url for c, _ in chunks))
        source_factor = min(unique_sources / 3, 1.0)  # Cap at 3 sources
        
        # Source diversity
        source_types = set(c.source for c, _ in chunks)
        diversity_factor = 1.0 if len(source_types) > 1 else 0.8
        
        # Content volume
        volume_factor = min(len(sentences) / 10, 1.0)
        
        confidence = (
            avg_relevance * 0.4 +
            source_factor * 0.25 +
            diversity_factor * 0.15 +
            volume_factor * 0.2
        )
        
        return round(min(confidence, 1.0), 3)


# Need numpy import at module level for TextRank
import numpy as np