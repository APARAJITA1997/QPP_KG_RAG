"""
Evaluation metrics for claim verification systems.

Metrics for retrieval quality and answer generation quality.
"""

import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Calculate retrieval quality metrics."""
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Recall@K: Proportion of relevant documents in top-k.
        
        Args:
            retrieved_ids: IDs of retrieved documents
            relevant_ids: IDs of relevant documents
            k: Cutoff position
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        num_relevant_retrieved = len(top_k_retrieved & relevant_set)
        num_relevant_total = len(relevant_set)
        
        recall = num_relevant_retrieved / num_relevant_total
        return float(recall)
    
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Precision@K: Proportion of top-k that are relevant.
        
        Args:
            retrieved_ids: IDs of retrieved documents
            relevant_ids: IDs of relevant documents
            k: Cutoff position
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_ids[:k]:
            return 0.0
        
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        num_relevant_retrieved = len(top_k_retrieved & relevant_set)
        precision = num_relevant_retrieved / k
        
        return float(precision)
    
    @staticmethod
    def mrr_at_k(
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: int = 10
    ) -> float:
        """
        Mean Reciprocal Rank@K: Position of first relevant document.
        
        Args:
            retrieved_ids: IDs of retrieved documents
            relevant_ids: IDs of relevant documents
            k: Cutoff position
            
        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        scores: List[float],
        relevant_ids: List[int],
        retrieved_ids: List[int],
        k: int = 10
    ) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        
        Args:
            scores: Relevance scores
            relevant_ids: IDs of relevant documents
            retrieved_ids: IDs of retrieved documents
            k: Cutoff position
            
        Returns:
            NDCG score (0-1)
        """
        # DCG
        dcg = 0.0
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = 1.0 if doc_id in relevant_set else 0.0
            dcg += relevance / np.log2(i + 2)
        
        # IDCG (ideal DCG with all relevant docs at top)
        idcg = 0.0
        for i in range(min(k, len(relevant_ids))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0.0:
            return 0.0
        
        ndcg = dcg / idcg
        return float(ndcg)
    
    @staticmethod
    def mean_reciprocal_rank(
        predictions: List[List[int]],
        ground_truth: List[List[int]],
        k: int = 10
    ) -> float:
        """
        Calculate MRR for multiple queries.
        
        Args:
            predictions: List of predicted document IDs
            ground_truth: List of relevant document ID lists
            k: Cutoff position
            
        Returns:
            Average MRR
        """
        mrrs = [
            RetrievalMetrics.mrr_at_k(pred, gt, k)
            for pred, gt in zip(predictions, ground_truth)
        ]
        return float(np.mean(mrrs))


class GenerationMetrics:
    """Calculate answer generation quality metrics."""
    
    @staticmethod
    def rouge_l(
        prediction: str,
        reference: str
    ) -> float:
        """
        ROUGE-L: F1 score of longest common subsequence.
        
        Args:
            prediction: Generated answer
            reference: Reference answer
            
        Returns:
            ROUGE-L F1 score (0-1)
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return float(scores['rougeL'].fmeasure)
    
    @staticmethod
    def rouge_1(
        prediction: str,
        reference: str
    ) -> float:
        """
        ROUGE-1: F1 score of unigram overlap.
        
        Args:
            prediction: Generated answer
            reference: Reference answer
            
        Returns:
            ROUGE-1 F1 score (0-1)
        """
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return float(scores['rouge1'].fmeasure)
    
    @staticmethod
    def exact_match(
        prediction: str,
        reference: str
    ) -> float:
        """
        Exact match: 1 if predictions equals reference, 0 otherwise.
        
        Args:
            prediction: Generated answer
            reference: Reference answer
            
        Returns:
            0.0 or 1.0
        """
        return 1.0 if prediction.strip() == reference.strip() else 0.0
    
    @staticmethod
    def f1_score(
        predicted_verdict: str,
        reference_verdict: str
    ) -> float:
        """
        F1 for stance/verdict prediction.
        
        Args:
            predicted_verdict: Predicted verdict (SUPPORTED/CONTRADICTED/NOINFO)
            reference_verdict: Reference verdict
            
        Returns:
            F1 score (0 or 1)
        """
        if predicted_verdict.strip() == reference_verdict.strip():
            return 1.0
        return 0.0


class EvaluationSuite:
    """
    Complete evaluation suite for claim verification systems.
    """
    
    @staticmethod
    def evaluate_retrieval(
        predictions: List[List[int]],
        ground_truth: List[List[int]],
        k_values: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            predictions: List of retrieved document ID lists
            ground_truth: List of relevant document ID lists
            k_values: Cutoff positions
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {}
        
        for k in k_values:
            mrrs = []
            recalls = []
            precisions = []
            
            for pred, gt in zip(predictions, ground_truth):
                mrr = RetrievalMetrics.mrr_at_k(pred, gt, k)
                recall = RetrievalMetrics.recall_at_k(pred, gt, k)
                precision = RetrievalMetrics.precision_at_k(pred, gt, k)
                
                mrrs.append(mrr)
                recalls.append(recall)
                precisions.append(precision)
            
            metrics[f'MRR@{k}'] = float(np.mean(mrrs))
            metrics[f'Recall@{k}'] = float(np.mean(recalls))
            metrics[f'Precision@{k}'] = float(np.mean(precisions))
        
        return metrics
    
    @staticmethod
    def evaluate_generation(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate answer generation.
        
        Args:
            predictions: List of generated answers
            references: List of reference answers
            
        Returns:
            Dictionary of generation metrics
        """
        rouge_ls = []
        rouge_1s = []
        ems = []
        
        for pred, ref in zip(predictions, references):
            rouge_l = GenerationMetrics.rouge_l(pred, ref)
            rouge_1 = GenerationMetrics.rouge_1(pred, ref)
            em = GenerationMetrics.exact_match(pred, ref)
            
            rouge_ls.append(rouge_l)
            rouge_1s.append(rouge_1)
            ems.append(em)
        
        metrics = {
            'ROUGE-L': float(np.mean(rouge_ls)),
            'ROUGE-1': float(np.mean(rouge_1s)),
            'Exact Match': float(np.mean(ems))
        }
        
        return metrics
    
    @staticmethod
    def format_results(results: Dict[str, float]) -> str:
        """Format results for display."""
        lines = []
        for metric, score in results.items():
            lines.append(f"  {metric}: {score:.4f}")
        return "\n".join(lines)


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example retrieval evaluation
    predictions = [[0, 1, 2, 3, 4], [0, 2, 3, 1, 4]]
    ground_truth = [[0, 2], [1, 3, 4]]
    
    retrieval_metrics = EvaluationSuite.evaluate_retrieval(
        predictions, ground_truth, k_values=[5]
    )
    print("Retrieval metrics:")
    print(EvaluationSuite.format_results(retrieval_metrics))
    
    # Example generation evaluation
    predictions = ["Insulin is produced by pancreas", "The pancreas makes insulin"]
    references = ["Insulin is a hormone from pancreas", "Pancreas produces insulin"]
    
    gen_metrics = EvaluationSuite.evaluate_generation(predictions, references)
    print("\nGeneration metrics:")
    print(EvaluationSuite.format_results(gen_metrics))
