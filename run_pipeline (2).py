"""
Complete QPP-KG-RAG pipeline for scientific claim verification.

Pipeline flow:
1. Load data
2. Initialize components
3. For each query:
   a. Retrieve passages
   b. Predict query difficulty (QPP)
   c. If Low-QPP: expand query using KG
   d. Re-rank results
   e. Generate answer
4. Evaluate results
"""

import logging
import yaml
from typing import List, Dict
import numpy as np

from src.data_loader import ClaimVerificationDataLoader
from src.preprocessing import TextPreprocessor
from src.knowledge_graph import KnowledgeGraphBuilder
from src.dense_retrieval import DenseRetriever
from src.difficulty_predictor import QueryPerformancePredictor
from src.query_expansion import KGAugmentedQueryExpander
from src.generation import EvidenceGroundedGenerator
from src.evaluation import EvaluationSuite

logger = logging.getLogger(__name__)


class AdaptiveClaimVerificationPipeline:
    """
    Complete adaptive pipeline for claim verification.
    
    Combines QPP, KG, and RAG into a unified system.
    """
    
    def __init__(self, config_path: str = 'config/base_config.yaml'):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Initializing components...")
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        
        self.retriever = DenseRetriever(
            model=self.config['models']['retrieval_model']
        )
        
        self.generator = EvidenceGroundedGenerator(
            model=self.config['models']['generation_model']
        )
        
        self.qpp_predictor = QueryPerformancePredictor(
            threshold=self.config['qpp']['threshold']
        )
        
        self.kg_builder = KnowledgeGraphBuilder()
        
        self.query_expander = KGAugmentedQueryExpander(
            kg_builder=self.kg_builder,
            model=self.config['models']['expansion_model']
        )
        
        logger.info("Components initialized")
    
    def process_query(
        self,
        query: str,
        use_qpp: bool = True,
        use_kg: bool = True
    ) -> Dict:
        """
        Process a single query through the full pipeline.
        
        Args:
            query: Query text
            use_qpp: Use QPP predictor
            use_kg: Use KG augmentation
            
        Returns:
            Dictionary with pipeline results
        """
        result = {'query': query}
        
        # Step 1: Retrieve initial passages
        retrieval_k = self.config['retrieval']['k']
        initial_results = self.retriever.retrieve(query, k=retrieval_k)
        
        if not initial_results:
            result['initial_passages'] = []
            result['initial_scores'] = []
            result['answer'] = "No passages found."
            return result
        
        initial_passages = [p for p, _, _ in initial_results]
        initial_scores = [s for _, s, _ in initial_results]
        result['initial_passages'] = initial_passages
        result['initial_scores'] = initial_scores
        
        # Step 2: Predict query difficulty (QPP)
        qpp_pred = None
        qpp_proba = None
        
        if use_qpp and self.qpp_predictor.model is not None:
            qpp_pred = self.qpp_predictor.predict(
                query, initial_scores, self.preprocessor
            )
            qpp_proba = self.qpp_predictor.predict_proba(
                query, initial_scores, self.preprocessor
            )
        
        result['qpp_prediction'] = qpp_pred
        result['qpp_probability'] = qpp_proba
        
        # Step 3: Expand query if Low-QPP
        final_passages = initial_passages
        final_scores = initial_scores
        
        if use_kg and use_qpp and qpp_pred == 0:  # Low QPP
            logger.info(f"Low-QPP detected. Expanding query...")
            
            # Get related entities
            expanded_query = self.query_expander.expand_query_simple(
                query, self.preprocessor
            )
            result['expanded_query'] = expanded_query
            
            # Retrieve with expanded query
            expanded_results = self.retriever.retrieve(
                expanded_query, k=retrieval_k
            )
            
            if expanded_results:
                expanded_passages = [p for p, _, _ in expanded_results]
                expanded_scores = [s for _, s, _ in expanded_results]
                
                # Step 4: Hybrid ranking
                alpha = self.config['hybrid_ranking']['alpha']
                beta = self.config['hybrid_ranking']['beta']
                
                # Normalize scores for combining
                initial_scores_norm = [s / max(initial_scores) for s in initial_scores] if max(initial_scores) > 0 else initial_scores
                expanded_scores_norm = [s / max(expanded_scores) for s in expanded_scores] if max(expanded_scores) > 0 else expanded_scores
                
                # Combine results - this is simplified
                # In practice, you'd merge passage lists and rerank
                final_passages = initial_passages + expanded_passages
                final_scores = initial_scores + expanded_scores
        
        # Step 5: Generate answer
        answer = self.generator.generate(query, final_passages[:5])
        
        result['final_passages'] = final_passages[:5]
        result['final_scores'] = final_scores[:5]
        result['answer'] = answer
        
        return result
    
    def process_batch(
        self,
        queries: List[str],
        use_qpp: bool = True,
        use_kg: bool = True
    ) -> List[Dict]:
        """
        Process multiple queries.
        
        Args:
            queries: List of query texts
            use_qpp: Use QPP predictor
            use_kg: Use KG augmentation
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, query in enumerate(queries):
            result = self.process_query(query, use_qpp, use_kg)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        return results


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = AdaptiveClaimVerificationPipeline('config/base_config.yaml')
    
    # Example queries
    queries = [
        "What is insulin?",
        "How does diabetes occur?"
    ]
    
    # Process queries
    results = pipeline.process_batch(queries)
    
    # Print results
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer']}")
