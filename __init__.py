"""
QPP-KG-RAG: Query Performance Prediction + Knowledge Graph + 
Retrieval-Augmented Generation for scientific claim verification.

Main modules:
- data_loader: Load claim verification datasets
- preprocessing: Text processing and entity extraction
- knowledge_graph: Build and query knowledge graphs
- dense_retrieval: Dense embedding-based retrieval
- difficulty_predictor: Query performance prediction
- query_expansion: Query expansion with KG augmentation
- generation: Evidence-grounded answer generation
- evaluation: Evaluation metrics
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Import main classes
from .data_loader import ClaimVerificationDataset, ClaimVerificationDataLoader
from .preprocessing import TextPreprocessor, EntityNormalizer
from .knowledge_graph import KnowledgeGraphBuilder
from .dense_retrieval import DenseRetriever
from .difficulty_predictor import QueryPerformancePredictor
from .query_expansion import KGAugmentedQueryExpander
from .generation import EvidenceGroundedGenerator
from .evaluation import RetrievalMetrics, GenerationMetrics, EvaluationSuite

__all__ = [
    'ClaimVerificationDataset',
    'ClaimVerificationDataLoader',
    'TextPreprocessor',
    'EntityNormalizer',
    'KnowledgeGraphBuilder',
    'DenseRetriever',
    'QueryPerformancePredictor',
    'KGAugmentedQueryExpander',
    'EvidenceGroundedGenerator',
    'RetrievalMetrics',
    'GenerationMetrics',
    'EvaluationSuite'
]
