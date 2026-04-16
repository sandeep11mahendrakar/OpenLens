# ============================================================
# evaluation/evaluator.py
# PURPOSE: Systematic evaluation of the full pipeline
# THIS IS WHAT SEPARATES HOBBYISTS FROM ENGINEERS
# ============================================================

import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class EvalCase:
    """A single evaluation test case."""
    query: str
    expected_mode: str
    expected_keywords: List[str]  # Words that should appear in the answer
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str    # Sub-category for detailed analysis


@dataclass  
class EvalResult:
    """Result of evaluating a single case."""
    query: str
    expected_mode: str
    predicted_mode: str
    mode_correct: bool
    mode_confidence: float
    keywords_found: List[str]
    keywords_missing: List[str]
    keyword_recall: float    # % of expected keywords found in answer
    answer_length: int       # Word count
    num_sources: int
    confidence: float
    latency_ms: float
    has_citations: bool


class PipelineEvaluator:
    """
    Comprehensive evaluation suite for the full pipeline.
    
    Metrics tracked:
    1. Intent Classification Accuracy
    2. Retrieval Quality (keyword recall as proxy)
    3. Answer Quality (length, citations, confidence)
    4. System Performance (latency, error rate)
    
    WHY THIS MATTERS:
    - Can't improve what you don't measure
    - Shows you understand ML evaluation methodology
    - Creates nice charts for your README
    """
    
    # Evaluation dataset
    EVAL_CASES = [
        # ADVICE - Easy
        EvalCase("How should I prepare for a coding interview?", "advice",
                ["practice", "algorithm", "interview"], "easy", "career"),
        EvalCase("Tips for saving money as a college student", "advice",
                ["budget", "save", "student"], "easy", "finance"),
        
        # ADVICE - Hard (could be confused with other modes)
        EvalCase("What's the best way to learn about history?", "advice",
                ["read", "learn", "book"], "hard", "education"),
        
        # HISTORY - Easy
        EvalCase("What caused World War I?", "history",
                ["assassination", "Franz Ferdinand", "alliance"], "easy", "war"),
        EvalCase("Tell me about the French Revolution", "history",
                ["France", "revolution", "monarchy"], "easy", "revolution"),
        
        # HISTORY - Hard
        EvalCase("How did we end up with the internet?", "history",
                ["ARPANET", "network", "computer"], "hard", "technology"),
        
        # SCIENCE - Easy
        EvalCase("How does photosynthesis work?", "science",
                ["light", "chlorophyll", "carbon dioxide", "plant"], "easy", "biology"),
        EvalCase("What causes lightning?", "science",
                ["electric", "charge", "cloud", "discharge"], "easy", "physics"),
        
        # SCIENCE - Hard
        EvalCase("Why is the sky blue?", "science",
                ["scatter", "light", "wavelength"], "medium", "physics"),
        
        # FACT - Easy
        EvalCase("What is the capital of France?", "fact",
                ["Paris"], "easy", "geography"),
        EvalCase("Who invented the telephone?", "fact",
                ["Bell", "telephone"], "easy", "invention"),
        
        # FACT - Hard (could be science or history)
        EvalCase("How far is the moon from Earth?", "fact",
                ["384", "km", "miles", "distance"], "medium", "astronomy"),
        
        # VERIFY - Easy
        EvalCase("Is it true that humans only use 10% of their brain?", "verify",
                ["myth", "false", "brain"], "easy", "health"),
        EvalCase("Can goldfish really only remember 3 seconds?", "verify",
                ["myth", "memory", "goldfish"], "easy", "animals"),
        
        # VERIFY - Hard
        EvalCase("Did Einstein really fail math?", "verify",
                ["Einstein", "math", "myth"], "hard", "education"),
    ]
    
    def __init__(self):
        # Import here to avoid circular imports
        from app.main import state
        self.state = state
    
    def run_full_evaluation(self) -> Dict:
        """Run all eval cases and compute aggregate metrics."""
        results = []
        errors = []
        
        print(f"\nRunning evaluation on {len(self.EVAL_CASES)} cases...")
        print("=" * 70)
        
        for i, case in enumerate(self.EVAL_CASES):
            try:
                result = self._evaluate_single(case)
                results.append(result)
                
                status = "✅" if result.mode_correct else "❌"
                print(f"  [{i+1}/{len(self.EVAL_CASES)}] {status} "
                      f"Mode: {result.predicted_mode:8s} "
                      f"(expected: {case.expected_mode:8s}) "
                      f"KW Recall: {result.keyword_recall:.0%} "
                      f"| {case.query[:50]}...")
                
            except Exception as e:
                errors.append({'case': case.query, 'error': str(e)})
                print(f"  [{i+1}/{len(self.EVAL_CASES)}] 💥 ERROR: {e}")
        
        # Compute aggregate metrics
        metrics = self._compute_metrics(results)
        metrics['errors'] = errors
        metrics['error_rate'] = len(errors) / len(self.EVAL_CASES)
        
        # Print summary
        self._print_summary(metrics)
        
        # Save results
        self._save_results(results, metrics)
        
        return metrics
    
    def _evaluate_single(self, case: EvalCase) -> EvalResult:
        """Evaluate a single test case."""
        start = time.time()
        
        # Classify intent
        intent_result = self.state.classifier.predict(case.query)
        
        # Run retrieval
        chunks = self.state.retrieval_pipeline.process_query(
            case.query, intent_result.predicted_mode, top_k=5
        )
        
        # Synthesize answer
        answer = self.state.synthesizer.synthesize(
            case.query, intent_result.predicted_mode, chunks
        )
        
        latency = (time.time() - start) * 1000
        
        # Check keyword recall
        answer_lower = answer.answer_text.lower()
        found = [kw for kw in case.expected_keywords if kw.lower() in answer_lower]
        missing = [kw for kw in case.expected_keywords if kw.lower() not in answer_lower]
        recall = len(found) / len(case.expected_keywords) if case.expected_keywords else 0
        
        return EvalResult(
            query=case.query,
            expected_mode=case.expected_mode,
            predicted_mode=intent_result.predicted_mode,
            mode_correct=intent_result.predicted_mode == case.expected_mode,
            mode_confidence=intent_result.confidence,
            keywords_found=found,
            keywords_missing=missing,
            keyword_recall=recall,
            answer_length=len(answer.answer_text.split()),
            num_sources=len(answer.citations),
            confidence=answer.confidence,
            latency_ms=latency,
            has_citations=len(answer.citations) > 0
        )
    
    def _compute_metrics(self, results: List[EvalResult]) -> Dict:
        """Compute aggregate metrics."""
        if not results:
            return {}
        
        mode_correct = [r.mode_correct for r in results]
        keyword_recalls = [r.keyword_recall for r in results]
        latencies = [r.latency_ms for r in results]
        confidences = [r.confidence for r in results]
        
        # Per-mode accuracy
        mode_accuracy = {}
        from collections import defaultdict
        mode_groups = defaultdict(list)
        for r in results:
            mode_groups[r.expected_mode].append(r.mode_correct)
        
        for mode, correct_list in mode_groups.items():
            mode_accuracy[mode] = sum(correct_list) / len(correct_list)
        
        return {
            'overall': {
                'intent_accuracy': sum(mode_correct) / len(mode_correct),
                'avg_keyword_recall': np.mean(keyword_recalls),
                'avg_confidence': np.mean(confidences),
                'avg_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'total_cases': len(results),
            },
            'per_mode_accuracy': mode_accuracy,
            'latency_distribution': {
                'min': min(latencies),
                'max': max(latencies),
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
            },
            'quality': {
                'avg_answer_length': np.mean([r.answer_length for r in results]),
                'avg_sources_per_answer': np.mean([r.num_sources for r in results]),
                'pct_with_citations': sum(r.has_citations for r in results) / len(results),
            }
        }
    
    def _print_summary(self, metrics: Dict):
        """Print a nice summary of evaluation results."""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        overall = metrics['overall']
        print(f"\n  Intent Classification Accuracy: {overall['intent_accuracy']:.1%}")
        print(f"  Average Keyword Recall:         {overall['avg_keyword_recall']:.1%}")
        print(f"  Average Confidence:             {overall['avg_confidence']:.3f}")
        print(f"  Average Latency:                {overall['avg_latency_ms']:.0f}ms")
        print(f"  P95 Latency:                    {metrics['latency_distribution']['p95']:.0f}ms")
        
        print(f"\n  Per-Mode Accuracy:")
        for mode, acc in metrics['per_mode_accuracy'].items():
            bar = "█" * int(acc * 20)
            print(f"    {mode:8s}: {acc:.1%} {bar}")
        
        quality = metrics['quality']
        print(f"\n  Answer Quality:")
        print(f"    Avg length:     {quality['avg_answer_length']:.0f} words")
        print(f"    Avg sources:    {quality['avg_sources_per_answer']:.1f}")
        print(f"    Has citations:  {quality['pct_with_citations']:.0%}")
        
        if metrics.get('errors'):
            print(f"\n  ⚠️  Errors: {len(metrics['errors'])}")
            for err in metrics['errors']:
                print(f"    - {err['case'][:40]}... → {err['error']}")
    
    def _save_results(self, results: List[EvalResult], metrics: Dict):
        """Save evaluation results for tracking over time."""
        output_dir = Path('evaluation/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_data = [asdict(r) for r in results]
        with open(output_dir / f'eval_{timestamp}_details.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save metrics summary
        with open(output_dir / f'eval_{timestamp}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\n  Results saved to {output_dir}/eval_{timestamp}_*")