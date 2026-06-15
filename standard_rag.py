"""
Standard RAG baseline without query performance prediction or KG augmentation.

Simply retrieves passages and generates answers.
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class StandardRAG:
    """
    Standard Retrieval-Augmented Generation baseline.
    
    Pipeline:
    1. Retrieve top-k passages using dense retrieval
    2. Generate answer from passages
    
    No query expansion or difficulty prediction.
    """
    
    def __init__(self, retriever, generator, device='cpu'):
        """
        Initialize RAG baseline.
        
        Args:
            retriever: DenseRetriever instance
            generator: EvidenceGroundedGenerator instance
            device: 'cpu' or 'cuda'
        """
        self.retriever = retriever
        self.generator = generator
        self.device = device
    
    def process_query(
        self,
        query: str,
        retrieval_k: int = 10
    ) -> Dict:
        """
        Process a single query.
        
        Args:
            query: Query text
            retrieval_k: Number of passages to retrieve
            
        Returns:
            Dictionary with results:
            - query: Original query
            - passages: Retrieved passages
            - scores: Retrieval scores
            - answer: Generated answer
        """
        # Step 1: Retrieve passages
        retrieval_results = self.retriever.retrieve(query, k=retrieval_k)
        
        if not retrieval_results:
            passages = []
            scores = []
            answer = "No relevant passages found."
        else:
            passages = [p for p, _, _ in retrieval_results]
            scores = [s for _, s, _ in retrieval_results]
            
            # Step 2: Generate answer
            answer = self.generator.generate(query, passages)
        
        result = {
            'query': query,
            'passages': passages,
            'scores': scores,
            'answer': answer,
            'num_passages_retrieved': len(passages)
        }
        
        return result
    
    def process_batch(
        self,
        queries: List[str],
        retrieval_k: int = 10
    ) -> List[Dict]:
        """
        Process multiple queries.
        
        Args:
            queries: List of query texts
            retrieval_k: Number of passages per query
            
        Returns:
            List of result dictionaries
        """
        results = []
        for i, query in enumerate(queries):
            result = self.process_query(query, retrieval_k)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        return results


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from src.dense_retrieval import DenseRetriever
    from src.generation import EvidenceGroundedGenerator
    
    # Initialize components
    retriever = DenseRetriever()
    generator = EvidenceGroundedGenerator()
    
    # Example passages (would normally be indexed)
    passages = [
        "Insulin is a hormone that regulates blood glucose.",
        "The pancreas produces insulin.",
        "Diabetes occurs when insulin production is impaired."
    ]
    retriever.index_passages(passages)
    
    # Create RAG baseline
    rag = StandardRAG(retriever, generator)
    
    # Process query
    result = rag.process_query("What is insulin?", retrieval_k=3)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
