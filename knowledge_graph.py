"""
Knowledge Graph construction and manipulation.

Features:
- Build KG from documents
- Query entity neighbors
- Find related concepts
"""

import networkx as nx
from typing import List, Dict, Set, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Build knowledge graphs from text documents.
    
    The graph is built by:
    1. Extracting entities from documents
    2. Creating edges between entities that co-occur
    3. Weighting edges by co-occurrence frequency
    
    Example:
        >>> kb = KnowledgeGraphBuilder()
        >>> texts = ["Insulin treats diabetes", "Pancreas produces insulin"]
        >>> kb.build_from_texts(texts)
        >>> neighbors = kb.get_neighbors('insulin', k=3)
        >>> print(neighbors)
    """
    
    def __init__(self):
        """Initialize knowledge graph builder."""
        self.graph = nx.Graph()
        self.entity_counts = defaultdict(int)
        self.edge_weights = defaultdict(int)
    
    def build_from_entities(
        self,
        entities_list: List[List[str]],
        min_cooccurrence: int = 2
    ) -> nx.Graph:
        """
        Build graph from lists of entities.
        
        Args:
            entities_list: List of entity lists 
                          (one list per document/passage)
            min_cooccurrence: Minimum co-occurrence count for edge creation
            
        Returns:
            NetworkX graph
        """
        # Count entity occurrences
        for entities in entities_list:
            for entity in set(entities):
                self.entity_counts[entity] += 1
        
        # Create edges for co-occurring entities
        for entities in entities_list:
            unique_entities = list(set(entities))
            for i, entity1 in enumerate(unique_entities):
                for entity2 in unique_entities[i+1:]:
                    edge_key = tuple(sorted([entity1, entity2]))
                    self.edge_weights[edge_key] += 1
        
        # Add nodes and edges to graph
        for entity in self.entity_counts:
            self.graph.add_node(entity)
        
        for (entity1, entity2), weight in self.edge_weights.items():
            if weight >= min_cooccurrence:
                self.graph.add_edge(entity1, entity2, weight=weight)
        
        logger.info(
            f"Built KG: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def build_from_texts(
        self,
        texts: List[str],
        preprocessor,
        min_cooccurrence: int = 2
    ) -> nx.Graph:
        """
        Build graph directly from texts.
        
        Args:
            texts: List of text documents
            preprocessor: TextPreprocessor instance
            min_cooccurrence: Minimum co-occurrence count
            
        Returns:
            NetworkX graph
        """
        # Extract entities from each text
        entities_list = [
            preprocessor.extract_entities(text)
            for text in texts
        ]
        
        # Build graph from entities
        return self.build_from_entities(entities_list, min_cooccurrence)
    
    def get_neighbors(
        self,
        entity: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get neighboring entities (related concepts).
        
        Args:
            entity: Query entity
            k: Number of neighbors to return
            
        Returns:
            List of (neighbor_entity, edge_weight) tuples, sorted by weight
        """
        if entity not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            weight = edge_data.get('weight', 1.0)
            neighbors.append((neighbor, weight))
        
        # Sort by weight (descending)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return neighbors[:k]
    
    def get_related_concepts(
        self,
        entities: List[str],
        k: int = 5
    ) -> List[str]:
        """
        Get all related concepts for a list of entities.
        
        Args:
            entities: List of query entities
            k: Neighbors per entity
            
        Returns:
            List of related entities
        """
        related = set()
        
        for entity in entities:
            neighbors = self.get_neighbors(entity, k=k)
            for neighbor, _ in neighbors:
                related.add(neighbor)
        
        return list(related)
    
    def shortest_path(
        self,
        entity1: str,
        entity2: str
    ) -> List[str]:
        """
        Find shortest path between two entities.
        
        Args:
            entity1: Start entity
            entity2: End entity
            
        Returns:
            List of entities forming the path
        """
        try:
            path = nx.shortest_path(self.graph, entity1, entity2)
            return path
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []
    
    def get_degree(self, entity: str) -> int:
        """
        Get the degree (number of connections) of an entity.
        
        Args:
            entity: Entity name
            
        Returns:
            Degree value
        """
        return self.graph.degree(entity) if entity in self.graph else 0
    
    def get_top_entities(self, k: int = 10) -> List[Tuple[str, int]]:
        """
        Get entities with highest degree.
        
        Args:
            k: Number of top entities
            
        Returns:
            List of (entity, degree) tuples
        """
        degrees = dict(self.graph.degree())
        sorted_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities[:k]
    
    def save_graph(self, filepath: str):
        """
        Save graph to file using pickle.
        
        Args:
            filepath: Output file path
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Saved KG to {filepath}")
    
    def load_graph(self, filepath: str):
        """
        Load graph from file.
        
        Args:
            filepath: Input file path
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Loaded KG from {filepath}")
    
    def graph_statistics(self) -> Dict:
        """
        Get basic graph statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph)
        }


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from preprocessing import TextPreprocessor
    
    # Create preprocessor and KG builder
    preprocessor = TextPreprocessor()
    kg_builder = KnowledgeGraphBuilder()
    
    # Example texts
    texts = [
        "Insulin is produced by the pancreas and treats diabetes.",
        "Diabetes is associated with obesity and poor diet.",
        "The pancreas is an important organ for glucose regulation.",
        "Insulin resistance leads to type 2 diabetes."
    ]
    
    # Build KG
    kg_builder.build_from_texts(texts, preprocessor, min_cooccurrence=1)
    
    # Query KG
    print("Graph statistics:", kg_builder.graph_statistics())
    print("Neighbors of 'insulin':", kg_builder.get_neighbors('insulin', k=3))
    print("Top entities:", kg_builder.get_top_entities(k=5))
