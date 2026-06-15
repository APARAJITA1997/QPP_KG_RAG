"""
Dense retrieval system using sentence embeddings and FAISS.

Features:
- Encode text to dense vectors
- Fast approximate nearest neighbor search
- Batch processing
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retrieval using pre-trained embeddings and FAISS index.
    
    Example:
        >>> retriever = DenseRetriever(model='sentence-transformers/all-MiniLM-L6-v2')
        >>> passages = ["Insulin treats diabetes", "Pancreas produces insulin"]
        >>> retriever.index_passages(passages)
        >>> query = "What treats diabetes?"
        >>> results = retriever.retrieve(query, k=1)
        >>> print(results)
    """
    
    def __init__(self, model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize dense retriever.
        
        Args:
            model: Sentence transformer model name
        """
        logger.info(f"Loading model: {model}")
        self.model = SentenceTransformer(model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.passages = []
        self.passage_embeddings = None
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_text(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text passages
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def index_passages(
        self,
        passages: List[str],
        batch_size: int = 32,
        use_gpu: bool = False
    ):
        """
        Index a collection of passages.
        
        Args:
            passages: List of text passages to index
            batch_size: Batch size for encoding
            use_gpu: Use GPU for FAISS index
        """
        logger.info(f"Indexing {len(passages)} passages...")
        
        self.passages = passages
        
        # Encode all passages
        self.passage_embeddings = self.encode_text(
            passages,
            batch_size=batch_size,
            show_progress=True
        )
        
        # Build FAISS index
        logger.info(f"Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.passage_embeddings / (
            np.linalg.norm(self.passage_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        self.index.add(normalized_embeddings.astype(np.float32))
        
        logger.info(f"Index built. Total passages: {len(self.passages)}")
    
    def retrieve(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Retrieve top-k passages for a query.
        
        Args:
            query: Query text
            k: Number of passages to retrieve
            
        Returns:
            List of (passage, score, passage_id) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_passages() first.")
        
        # Encode query
        query_embedding = self.encode_text([query], show_progress=False)[0]
        
        # Normalize for cosine similarity
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Search index
        scores, passage_ids = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k=min(k, len(self.passages))
        )
        
        results = []
        for score, passage_id in zip(scores[0], passage_ids[0]):
            if passage_id == -1:  # Invalid ID
                continue
            results.append((
                self.passages[int(passage_id)],
                float(score),
                int(passage_id)
            ))
        
        return results
    
    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 10
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Retrieve passages for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of passages per query
            
        Returns:
            List of result lists
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_passages() first.")
        
        # Encode queries
        query_embeddings = self.encode_text(
            queries,
            batch_size=32,
            show_progress=False
        )
        
        # Normalize
        query_embeddings = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Search
        scores, passage_ids = self.index.search(
            query_embeddings.astype(np.float32),
            k=min(k, len(self.passages))
        )
        
        # Format results
        all_results = []
        for query_scores, query_ids in zip(scores, passage_ids):
            results = []
            for score, passage_id in zip(query_scores, query_ids):
                if passage_id == -1:
                    continue
                results.append((
                    self.passages[int(passage_id)],
                    float(score),
                    int(passage_id)
                ))
            all_results.append(results)
        
        return all_results
    
    def get_passage_embedding(self, passage_id: int) -> np.ndarray:
        """
        Get embedding for a passage.
        
        Args:
            passage_id: Passage ID
            
        Returns:
            Embedding vector
        """
        if self.passage_embeddings is None:
            raise RuntimeError("Passages not indexed.")
        return self.passage_embeddings[passage_id]
    
    def get_passage(self, passage_id: int) -> str:
        """
        Get passage text by ID.
        
        Args:
            passage_id: Passage ID
            
        Returns:
            Passage text
        """
        return self.passages[passage_id]
    
    def compute_similarity(
        self,
        query: str,
        passage: str
    ) -> float:
        """
        Compute similarity between query and passage.
        
        Args:
            query: Query text
            passage: Passage text
            
        Returns:
            Similarity score (0-1)
        """
        query_emb = self.encode_text([query])[0]
        passage_emb = self.encode_text([passage])[0]
        
        # Cosine similarity
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        passage_emb = passage_emb / (np.linalg.norm(passage_emb) + 1e-8)
        
        similarity = np.dot(query_emb, passage_emb)
        return float(similarity)
    
    def save_index(self, filepath: str):
        """
        Save index and passages to file.
        
        Args:
            filepath: Output file path
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'passages': self.passages,
                'embeddings': self.passage_embeddings
            }, f)
        logger.info(f"Saved index to {filepath}")
    
    def load_index(self, filepath: str):
        """
        Load index and passages from file.
        
        Args:
            filepath: Input file path
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.index = data['index']
        self.passages = data['passages']
        self.passage_embeddings = data['embeddings']
        
        logger.info(f"Loaded index from {filepath}")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retriever
    retriever = DenseRetriever()
    
    # Example passages
    passages = [
        "Insulin is a hormone produced by the pancreas.",
        "Diabetes is a metabolic disease.",
        "The pancreas regulates blood glucose levels.",
        "Insulin resistance is associated with obesity."
    ]
    
    # Index passages
    retriever.index_passages(passages)
    
    # Retrieve
    query = "What hormone regulates blood sugar?"
    results = retriever.retrieve(query, k=2)
    
    print(f"Query: {query}")
    for passage, score, passage_id in results:
        print(f"  Score: {score:.3f} | {passage}")
