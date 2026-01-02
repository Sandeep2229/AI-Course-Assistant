"""
Retrieval Evaluation Framework for RAG System.

Implements standard IR metrics to measure retrieval quality:
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that are retrieved  
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- Hit Rate@K: Whether any relevant doc appears in top K

This differentiates production RAG systems from toy projects.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Single evaluation result."""
    query: str
    expected_sources: List[str]
    retrieved_sources: List[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool
    latency_ms: float


@dataclass 
class EvalSummary:
    """Aggregated evaluation metrics."""
    num_queries: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    timestamp: str


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.
    
    Usage:
        evaluator = RetrievalEvaluator(retriever)
        evaluator.add_test_case("What is the grading policy?", ["syllabus.pdf"])
        summary = evaluator.run_evaluation(k=5)
        print(f"Precision@5: {summary.precision_at_k:.2%}")
    """
    
    def __init__(self, retriever, k: int = 5):
        """
        Initialize evaluator with a retriever instance.
        
        Args:
            retriever: CourseRetriever instance
            k: Default number of documents to retrieve
        """
        self.retriever = retriever
        self.k = k
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[EvalResult] = []
        logger.info(f"RetrievalEvaluator initialized with k={k}")
    
    def add_test_case(
        self,
        query: str,
        expected_sources: List[str],
        course_id: Optional[str] = None
    ) -> None:
        """
        Add a test case for evaluation.
        
        Args:
            query: Test query
            expected_sources: List of source files that should be retrieved
            course_id: Optional course filter
        """
        self.test_cases.append({
            "query": query,
            "expected_sources": expected_sources,
            "course_id": course_id
        })
        logger.debug(f"Added test case: {query[:50]}...")
    
    def load_test_cases(self, filepath: str) -> int:
        """
        Load test cases from JSON file.
        
        Expected format:
        [
            {"query": "...", "expected_sources": ["file1.pdf"], "course_id": "CS101"},
            ...
        ]
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Number of test cases loaded
        """
        with open(filepath, 'r') as f:
            cases = json.load(f)
        
        for case in cases:
            self.add_test_case(
                query=case["query"],
                expected_sources=case["expected_sources"],
                course_id=case.get("course_id")
            )
        
        logger.info(f"Loaded {len(cases)} test cases from {filepath}")
        return len(cases)
    
    def save_test_cases(self, filepath: str) -> None:
        """Save current test cases to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.test_cases, f, indent=2)
        logger.info(f"Saved {len(self.test_cases)} test cases to {filepath}")
    
    def _calculate_precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / k
    
    def _calculate_recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if len(relevant) == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    def _calculate_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Reciprocal Rank (1/rank of first relevant result)."""
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0
    
    def evaluate_single(
        self,
        query: str,
        expected_sources: List[str],
        course_id: Optional[str] = None,
        k: Optional[int] = None
    ) -> EvalResult:
        """
        Evaluate a single query.
        
        Args:
            query: Test query
            expected_sources: Expected source files
            course_id: Optional course filter
            k: Number of docs to retrieve (default: self.k)
            
        Returns:
            EvalResult with metrics
        """
        k = k or self.k
        
        # Time the retrieval
        start_time = time.perf_counter()
        documents = self.retriever.retrieve(
            query=query,
            course_id=course_id,
            k=k
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract source files from retrieved documents
        retrieved_sources = [
            doc.metadata.get('source_file', 'Unknown')
            for doc in documents
        ]
        
        # Calculate metrics
        precision = self._calculate_precision_at_k(retrieved_sources, expected_sources, k)
        recall = self._calculate_recall_at_k(retrieved_sources, expected_sources, k)
        rr = self._calculate_reciprocal_rank(retrieved_sources, expected_sources)
        hit = any(src in expected_sources for src in retrieved_sources[:k])
        
        return EvalResult(
            query=query,
            expected_sources=expected_sources,
            retrieved_sources=retrieved_sources,
            precision_at_k=precision,
            recall_at_k=recall,
            reciprocal_rank=rr,
            hit=hit,
            latency_ms=latency_ms
        )
    
    def run_evaluation(
        self,
        k: Optional[int] = None,
        verbose: bool = True
    ) -> EvalSummary:
        """
        Run evaluation on all test cases.
        
        Args:
            k: Number of docs to retrieve (default: self.k)
            verbose: Print progress and results
            
        Returns:
            EvalSummary with aggregated metrics
        """
        if not self.test_cases:
            raise ValueError("No test cases added. Use add_test_case() or load_test_cases() first.")
        
        k = k or self.k
        self.results = []
        latencies = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Retrieval Evaluation (k={k})")
            print(f"{'='*60}")
        
        for i, case in enumerate(self.test_cases, 1):
            result = self.evaluate_single(
                query=case["query"],
                expected_sources=case["expected_sources"],
                course_id=case.get("course_id"),
                k=k
            )
            self.results.append(result)
            latencies.append(result.latency_ms)
            
            if verbose:
                status = "✓" if result.hit else "✗"
                print(f"  [{i}/{len(self.test_cases)}] {status} P@{k}={result.precision_at_k:.2f} "
                      f"R@{k}={result.recall_at_k:.2f} RR={result.reciprocal_rank:.2f} "
                      f"({result.latency_ms:.0f}ms)")
        
        # Calculate aggregated metrics
        n = len(self.results)
        summary = EvalSummary(
            num_queries=n,
            precision_at_k=sum(r.precision_at_k for r in self.results) / n,
            recall_at_k=sum(r.recall_at_k for r in self.results) / n,
            mrr=sum(r.reciprocal_rank for r in self.results) / n,
            hit_rate=sum(r.hit for r in self.results) / n,
            avg_latency_ms=sum(latencies) / n,
            p95_latency_ms=sorted(latencies)[int(0.95 * n)] if n > 1 else latencies[0],
            timestamp=datetime.now().isoformat()
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(f"  Queries evaluated: {summary.num_queries}")
            print(f"  Precision@{k}:      {summary.precision_at_k:.2%}")
            print(f"  Recall@{k}:         {summary.recall_at_k:.2%}")
            print(f"  MRR:                {summary.mrr:.3f}")
            print(f"  Hit Rate@{k}:       {summary.hit_rate:.2%}")
            print(f"  Avg Latency:        {summary.avg_latency_ms:.0f}ms")
            print(f"  P95 Latency:        {summary.p95_latency_ms:.0f}ms")
            print(f"{'='*60}\n")
        
        return summary
    
    def export_results(self, filepath: str) -> None:
        """Export detailed results to JSON."""
        if not self.results:
            raise ValueError("No results to export. Run evaluation first.")
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "k": self.k,
            "num_queries": len(self.results),
            "results": [
                {
                    "query": r.query,
                    "expected_sources": r.expected_sources,
                    "retrieved_sources": r.retrieved_sources,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                    "hit": r.hit,
                    "latency_ms": r.latency_ms
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.results)} results to {filepath}")


def create_sample_test_cases() -> List[Dict[str, Any]]:
    """
    Create sample test cases template.
    
    Returns a template that users should customize based on their uploaded documents.
    """
    return [
        {
            "query": "What is the grading policy for this course?",
            "expected_sources": ["syllabus.pdf"],
            "course_id": None
        },
        {
            "query": "When is the midterm exam scheduled?",
            "expected_sources": ["syllabus.pdf", "schedule.pdf"],
            "course_id": None
        },
        {
            "query": "What are the office hours?",
            "expected_sources": ["syllabus.pdf"],
            "course_id": None
        },
        {
            "query": "What topics are covered in week 3?",
            "expected_sources": ["schedule.pdf", "lecture_03.pdf"],
            "course_id": None
        },
        {
            "query": "What is the late submission policy?",
            "expected_sources": ["syllabus.pdf"],
            "course_id": None
        }
    ]


# CLI for running evaluation
if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.vector_store import VectorStoreManager
    from src.retriever import CourseRetriever
    
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--test-file", type=str, help="Path to test cases JSON")
    parser.add_argument("--k", type=int, default=5, help="Number of docs to retrieve")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--generate-template", type=str, help="Generate sample test cases template")
    
    args = parser.parse_args()
    
    if args.generate_template:
        template = create_sample_test_cases()
        with open(args.generate_template, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"Generated template at {args.generate_template}")
        print("Edit this file with queries and expected sources based on your documents.")
        sys.exit(0)
    
    # Initialize components
    print("Initializing retrieval system...")
    vs = VectorStoreManager()
    retriever = CourseRetriever(vs)
    evaluator = RetrievalEvaluator(retriever, k=args.k)
    
    # Load test cases
    if args.test_file:
        evaluator.load_test_cases(args.test_file)
    else:
        print("No test file provided. Using sample test cases.")
        for case in create_sample_test_cases():
            evaluator.add_test_case(**case)
    
    # Run evaluation
    summary = evaluator.run_evaluation(k=args.k, verbose=True)
    
    # Export if requested
    if args.export:
        evaluator.export_results(args.export)
        print(f"Results exported to {args.export}")

