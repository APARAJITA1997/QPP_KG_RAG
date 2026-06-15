"""
Query Performance Prediction (QPP) module.

Predicts whether a query will retrieve good or poor results
before executing the actual retrieval.

Features:
- Extract QPP features from queries
- Train XGBoost classifier
- Predict query difficulty
"""

import numpy as np
import xgboost as xgb
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryPerformancePredictor:
    """
    Predict query difficulty before retrieval.
    
    Uses 4 features:
    1. Query Length: Number of tokens
    2. Term Specificity: Rarity of terms  
    3. Relevance Score: Average similarity of top-k results
    4. Score Dispersion: Spread of top-k scores
    
    Example:
        >>> predictor = QueryPerformancePredictor()
        >>> predictor.train(X_train, y_train)
        >>> qpp_score = predictor.predict("What treats diabetes?")
        >>> print(qpp_score)  # Low (0) = poor, High (1) = good
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize predictor.
        
        Args:
            threshold: Classification threshold for High/Low QPP
        """
        self.model = None
        self.threshold = threshold
        self.feature_names = [
            'query_length',
            'term_specificity',
            'relevance_score',
            'score_dispersion'
        ]
        logger.info(f"QPP threshold: {threshold}")
    
    def extract_features(
        self,
        query: str,
        retrieval_scores: List[float],
        preprocessor
    ) -> Dict[str, float]:
        """
        Extract QPP features from query and retrieval results.
        
        Args:
            query: Query text
            retrieval_scores: Similarity scores of top-k retrieval results
            preprocessor: TextPreprocessor instance
            
        Returns:
            Dictionary of features
        """
        # Feature 1: Query length
        query_length = len(preprocessor.get_tokens(query))
        
        # Feature 2: Term specificity
        term_specificity = preprocessor.calculate_term_specificity(query)
        
        # Feature 3: Relevance score (average of top-k)
        relevance_score = np.mean(retrieval_scores) if retrieval_scores else 0.0
        
        # Feature 4: Score dispersion (standard deviation)
        score_dispersion = np.std(retrieval_scores) if len(retrieval_scores) > 1 else 0.0
        
        features = {
            'query_length': float(query_length),
            'term_specificity': float(term_specificity),
            'relevance_score': float(relevance_score),
            'score_dispersion': float(score_dispersion)
        }
        
        return features
    
    def extract_features_batch(
        self,
        queries: List[str],
        retrieval_scores_list: List[List[float]],
        preprocessor
    ) -> List[Dict[str, float]]:
        """
        Extract features for multiple queries.
        
        Args:
            queries: List of query texts
            retrieval_scores_list: List of score lists
            preprocessor: TextPreprocessor instance
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        for query, scores in zip(queries, retrieval_scores_list):
            features = self.extract_features(query, scores, preprocessor)
            features_list.append(features)
        return features_list
    
    def _features_to_array(
        self,
        features_list: List[Dict[str, float]]
    ) -> np.ndarray:
        """Convert feature dicts to numpy array."""
        X = []
        for features in features_list:
            row = [
                features['query_length'],
                features['term_specificity'],
                features['relevance_score'],
                features['score_dispersion']
            ]
            X.append(row)
        return np.array(X, dtype=np.float32)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        max_depth: int = 5,
        n_estimators: int = 100
    ):
        """
        Train XGBoost classifier.
        
        Args:
            X: Feature array (n_samples, 4)
            y: Labels (0=Low QPP, 1=High QPP)
            test_size: Validation set fraction
            max_depth: Tree depth
            n_estimators: Number of boosting rounds
        """
        logger.info(f"Training QPP model with {len(X)} samples...")
        
        # Split into train/validation
        n_train = int(len(X) * (1 - test_size))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Train model
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        logger.info(f"Train accuracy: {train_acc:.3f}")
        logger.info(f"Validation accuracy: {val_acc:.3f}")
    
    def predict(
        self,
        query: str,
        retrieval_scores: List[float],
        preprocessor
    ) -> int:
        """
        Predict query difficulty.
        
        Args:
            query: Query text
            retrieval_scores: Top-k retrieval scores
            preprocessor: TextPreprocessor instance
            
        Returns:
            0 (Low QPP - poor retrieval expected)
            1 (High QPP - good retrieval expected)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features = self.extract_features(query, retrieval_scores, preprocessor)
        X = self._features_to_array([features])
        
        prediction = self.model.predict(X)[0]
        return int(prediction)
    
    def predict_proba(
        self,
        query: str,
        retrieval_scores: List[float],
        preprocessor
    ) -> float:
        """
        Get probability of High QPP.
        
        Args:
            query: Query text
            retrieval_scores: Top-k retrieval scores
            preprocessor: TextPreprocessor instance
            
        Returns:
            Probability in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features = self.extract_features(query, retrieval_scores, preprocessor)
        X = self._features_to_array([features])
        
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)
    
    def predict_batch(
        self,
        queries: List[str],
        retrieval_scores_list: List[List[float]],
        preprocessor
    ) -> List[int]:
        """
        Predict for multiple queries.
        
        Args:
            queries: List of query texts
            retrieval_scores_list: List of score lists
            preprocessor: TextPreprocessor instance
            
        Returns:
            List of predictions (0 or 1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features_list = self.extract_features_batch(
            queries, retrieval_scores_list, preprocessor
        )
        X = self._features_to_array(features_list)
        
        predictions = self.model.predict(X)
        return [int(p) for p in predictions]
    
    def predict_proba_batch(
        self,
        queries: List[str],
        retrieval_scores_list: List[List[float]],
        preprocessor
    ) -> List[float]:
        """
        Get probabilities for multiple queries.
        
        Args:
            queries: List of query texts
            retrieval_scores_list: List of score lists
            preprocessor: TextPreprocessor instance
            
        Returns:
            List of probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features_list = self.extract_features_batch(
            queries, retrieval_scores_list, preprocessor
        )
        X = self._features_to_array(features_list)
        
        probas = self.model.predict_proba(X)[:, 1]
        return [float(p) for p in probas]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        importances = self.model.feature_importances_
        importance_dict = {
            name: float(score)
            for name, score in zip(self.feature_names, importances)
        }
        return importance_dict
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Output file path
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        self.model.save_model(filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Input file path
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        logger.info(f"Loaded model from {filepath}")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from preprocessing import TextPreprocessor
    
    # Create predictor
    predictor = QueryPerformancePredictor(threshold=0.5)
    preprocessor = TextPreprocessor()
    
    # Example training data
    # Query 1: specific, should be High QPP
    queries = ["What hormone regulates blood sugar?"]
    scores_list = [[0.8, 0.75, 0.72, 0.68]]
    
    # Query 2: vague, should be Low QPP
    queries.append("What are the effects of diet?")
    scores_list.append([[0.45, 0.42, 0.38, 0.35]])
    
    # Create dummy training set
    X_train = np.array([
        [4, 0.8, 0.75, 0.05],  # High QPP
        [5, 0.6, 0.42, 0.08],  # Low QPP
        [3, 0.9, 0.82, 0.03],  # High QPP
        [6, 0.4, 0.35, 0.10],  # Low QPP
    ], dtype=np.float32)
    
    y_train = np.array([1, 0, 1, 0])
    
    # Train
    predictor.train(X_train, y_train)
    
    # Predict
    for query, scores in zip(queries, scores_list):
        pred = predictor.predict(query, scores, preprocessor)
        proba = predictor.predict_proba(query, scores, preprocessor)
        print(f"Query: {query}")
        print(f"  Prediction: {'High QPP' if pred == 1 else 'Low QPP'}")
        print(f"  Probability: {proba:.3f}")
