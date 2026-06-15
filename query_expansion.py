"""
Query expansion using knowledge graph augmentation.

Expands difficult queries by adding related concepts from KG.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class KGAugmentedQueryExpander:
    """
    Expand queries using knowledge graph neighbors.
    
    For Low-QPP queries, adds related concepts from KG
    to make the query more specific and easier to retrieve for.
    
    Example:
        >>> expander = KGAugmentedQueryExpander(kg_builder=kg_builder)
        >>> original = "effects of bad diet"
        >>> expanded = expander.expand_query(original, preprocessor)
        >>> print(expanded)
        "effects of bad diet obesity diabetes heart disease"
    """
    
    def __init__(
        self,
        kg_builder,
        model: str = 't5-small',
        device: str = 'cpu'
    ):
        """
        Initialize query expander.
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance
            model: T5 model for rewriting
            device: 'cpu' or 'cuda'
        """
        self.kg_builder = kg_builder
        self.device = device
        
        logger.info(f"Loading model: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
    
    def get_related_entities(
        self,
        query: str,
        preprocessor,
        k_neighbors: int = 5
    ) -> List[str]:
        """
        Get entities related to query entities.
        
        Args:
            query: Query text
            preprocessor: TextPreprocessor instance
            k_neighbors: Number of neighbors per entity
            
        Returns:
            List of related entity names
        """
        # Extract entities from query
        entities = preprocessor.extract_entities(query)
        
        if not entities:
            return []
        
        # Get KG neighbors for each entity
        related = set()
        for entity in entities:
            neighbors = self.kg_builder.get_neighbors(entity, k=k_neighbors)
            for neighbor, _ in neighbors:
                related.add(neighbor)
        
        return list(related)
    
    def expand_query_simple(
        self,
        query: str,
        preprocessor,
        k_neighbors: int = 5,
        max_additions: int = 5
    ) -> str:
        """
        Simple query expansion: append related entities.
        
        Args:
            query: Original query
            preprocessor: TextPreprocessor instance
            k_neighbors: Neighbors per entity
            max_additions: Max entities to add
            
        Returns:
            Expanded query
        """
        related_entities = self.get_related_entities(
            query, preprocessor, k_neighbors
        )
        
        # Take only top additions
        additions = related_entities[:max_additions]
        
        # Append to query
        if additions:
            expanded = f"{query} {' '.join(additions)}"
        else:
            expanded = query
        
        return expanded
    
    def expand_query_with_t5(
        self,
        query: str,
        preprocessor,
        k_neighbors: int = 5,
        max_additions: int = 5
    ) -> str:
        """
        Expand query using T5 rewriting.
        
        Uses T5 to naturally incorporate related entities.
        
        Args:
            query: Original query
            preprocessor: TextPreprocessor instance
            k_neighbors: Neighbors per entity
            max_additions: Max entities to add
            
        Returns:
            Expanded query
        """
        # Get related entities
        related_entities = self.get_related_entities(
            query, preprocessor, k_neighbors
        )
        
        if not related_entities:
            return query
        
        # Take only top additions
        additions = related_entities[:max_additions]
        additions_text = ", ".join(additions)
        
        # Create prompt for T5
        prompt = f"expand query: {query}. Related concepts: {additions_text}"
        
        # Encode
        inputs = self.tokenizer(
            prompt,
            max_length=256,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=1,
                early_stopping=True
            )
        
        # Decode
        expanded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return expanded
    
    def expand_batch(
        self,
        queries: List[str],
        preprocessor,
        k_neighbors: int = 5,
        use_t5: bool = False
    ) -> List[str]:
        """
        Expand multiple queries.
        
        Args:
            queries: List of query texts
            preprocessor: TextPreprocessor instance
            k_neighbors: Neighbors per entity
            use_t5: Use T5 for rewriting
            
        Returns:
            List of expanded queries
        """
        expanded_queries = []
        
        for query in queries:
            if use_t5:
                expanded = self.expand_query_with_t5(
                    query, preprocessor, k_neighbors
                )
            else:
                expanded = self.expand_query_simple(
                    query, preprocessor, k_neighbors
                )
            expanded_queries.append(expanded)
        
        return expanded_queries
    
    def get_expansion_info(
        self,
        query: str,
        preprocessor,
        k_neighbors: int = 5
    ) -> Dict:
        """
        Get detailed expansion information.
        
        Args:
            query: Query text
            preprocessor: TextPreprocessor instance
            k_neighbors: Neighbors per entity
            
        Returns:
            Dictionary with expansion details
        """
        # Extract entities
        entities = preprocessor.extract_entities(query)
        
        # Get neighbors for each entity
        entity_neighbors = {}
        for entity in entities:
            neighbors = self.kg_builder.get_neighbors(entity, k=k_neighbors)
            entity_neighbors[entity] = [n for n, _ in neighbors]
        
        # Expanded queries
        simple_expanded = self.expand_query_simple(
            query, preprocessor, k_neighbors
        )
        
        info = {
            'original_query': query,
            'entities': entities,
            'entity_neighbors': entity_neighbors,
            'simple_expansion': simple_expanded,
            'num_related_entities': len(self.get_related_entities(
                query, preprocessor, k_neighbors
            ))
        }
        
        return info


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from preprocessing import TextPreprocessor
    from knowledge_graph import KnowledgeGraphBuilder
    
    # Setup
    preprocessor = TextPreprocessor()
    kg_builder = KnowledgeGraphBuilder()
    
    # Build KG
    texts = [
        "Diet affects obesity and diabetes",
        "Poor nutrition causes heart disease",
        "Insulin resistance and obesity are related",
    ]
    kg_builder.build_from_texts(texts, preprocessor, min_cooccurrence=1)
    
    # Expand query
    expander = KGAugmentedQueryExpander(kg_builder)
    query = "effects of diet"
    expanded = expander.expand_query_simple(query, preprocessor)
    
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")
