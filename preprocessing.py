"""
Text preprocessing and entity extraction module.

Features:
- Text normalization
- Named entity recognition (NER)
- Entity linking
- Token analysis
"""

import spacy
import re
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing and entity extraction.
    
    Example:
        >>> preprocessor = TextPreprocessor()
        >>> text = "Insulin is produced by the pancreas."
        >>> entities = preprocessor.extract_entities(text)
        >>> print(entities)
        ['Insulin', 'pancreas']
    """
    
    def __init__(self, model: str = 'en_core_web_sm'):
        """
        Initialize preprocessor.
        
        Args:
            model: spaCy model name
        """
        try:
            self.nlp = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
        except OSError:
            logger.error(f"Model {model} not found. Installing...")
            import os
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove extra spaces.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep alphanumeric, spaces, hyphens)
        text = re.sub(r'[^\w\s\-]', '', text)
        
        return text.strip()
    
    def extract_entities(
        self, 
        text: str,
        entity_types: List[str] = None
    ) -> List[str]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            entity_types: Filter by entity types 
                         (e.g., ['PERSON', 'ORG', 'GPE'])
                         If None, returns all entities
            
        Returns:
            List of entity texts
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                entities.append(ent.text)
        
        return list(set(entities))  # Remove duplicates
    
    def extract_entity_dict(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities with their types.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to entity lists
        """
        doc = self.nlp(text)
        entities_by_type = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities_by_type:
                entities_by_type[ent.label_] = []
            entities_by_type[ent.label_].append(ent.text)
        
        # Remove duplicates within each type
        for entity_type in entities_by_type:
            entities_by_type[entity_type] = list(
                set(entities_by_type[entity_type])
            )
        
        return entities_by_type
    
    def get_tokens(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def get_lemmas(self, text: str) -> List[str]:
        """
        Get lemmatized tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of lemmas
        """
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags.
        
        Args:
            text: Input text
            
        Returns:
            List of (token, POS_tag) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def calculate_term_specificity(self, text: str, corpus_size: int = 1000000) -> float:
        """
        Calculate term specificity (IDF-like metric).
        
        Args:
            text: Input text
            corpus_size: Total corpus size
            
        Returns:
            Specificity score (0-1)
        """
        tokens = self.get_tokens(text)
        if not tokens:
            return 0.0
        
        # Simple heuristic: rare words have higher specificity
        # In practice, use actual IDF scores from your corpus
        specificity = len(set(tokens)) / len(tokens)
        return specificity
    
    def extract_query_features(self, query: str) -> Dict[str, float]:
        """
        Extract features from a query.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary with query features:
            - query_length: Number of tokens
            - term_specificity: Rareness of terms
            - avg_token_length: Average token length
        """
        tokens = self.get_tokens(query)
        
        features = {
            'query_length': len(tokens),
            'term_specificity': self.calculate_term_specificity(query),
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0,
            'num_entities': len(self.extract_entities(query))
        }
        
        return features


class EntityNormalizer:
    """Normalize and standardize entity mentions."""
    
    @staticmethod
    def normalize_entity(entity: str) -> str:
        """
        Normalize an entity mention.
        
        Args:
            entity: Entity text
            
        Returns:
            Normalized entity
        """
        # Lowercase
        entity = entity.lower()
        
        # Remove articles
        entity = re.sub(r'\b(a|an|the)\b\s*', '', entity)
        
        # Remove extra spaces
        entity = re.sub(r'\s+', ' ', entity).strip()
        
        return entity
    
    @staticmethod
    def is_same_entity(entity1: str, entity2: str) -> bool:
        """
        Check if two entities refer to the same thing.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities are the same, False otherwise
        """
        norm1 = EntityNormalizer.normalize_entity(entity1)
        norm2 = EntityNormalizer.normalize_entity(entity2)
        
        return norm1 == norm2


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = TextPreprocessor()
    
    # Example text
    text = "Insulin is a hormone produced by the pancreas. It regulates blood sugar levels."
    
    print(f"Original text: {text}")
    print(f"Normalized: {preprocessor.normalize_text(text)}")
    print(f"Entities: {preprocessor.extract_entities(text)}")
    print(f"Entities with types: {preprocessor.extract_entity_dict(text)}")
    print(f"Query features: {preprocessor.extract_query_features(text)}")
