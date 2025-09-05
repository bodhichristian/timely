"""
Smart issue triage: returns up to 3 category tags with confidence scores.
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from model_utils import load_model_artifacts
from feature_engineering import TextFeatureExtractor

class SmartIssueTriage:
    def __init__(self, model_dir: str):
        """
        Initialize the smart issue triage system
        
        Args:
            model_dir: Directory containing saved model artifacts
        """
        # Load artifacts
        self.artifacts = load_model_artifacts(model_dir)
        self.model = self.artifacts['model']
        self.tfidf_vectorizer = self.artifacts['tfidf_vectorizer']
        self.label_encoder = self.artifacts['label_encoder']
        self.repo_encoder = self.artifacts['repo_encoder']
        
        # Initialize feature extractor
        self.feature_extractor = TextFeatureExtractor(self.tfidf_vectorizer)
        
        # No opinionated thresholds here; UI controls the minimum confidence
    
    def get_recommendations(
        self,
        features: pd.DataFrame,
        threshold: float = 0.30
    ) -> Dict:
        """
        Return up to 3 suggested category tags with confidence.
        """
        proba = self.model.predict_proba(features)[0]
        classes = self.label_encoder.inverse_transform(range(len(proba)))
        ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
        # Always return top 3; UI decides what to display based on threshold
        suggestions = [
            {'tag': cat, 'confidence': float(conf)}
            for cat, conf in ranked[:3]
        ]
        return {'suggested_tags': suggestions}
    
    def predict(
        self,
        title: str,
        body: str,
        repo: str,
        threshold: float = 0.30
    ) -> Dict:
        """Predict and return up to 3 tags with confidence."""
        # Combine text
        text = f"{title}\n{body}"
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            text=text,
            repo=repo,
            repo_encoder=self.repo_encoder
        )
        return self.get_recommendations(features=features, threshold=threshold)
    
    def batch_predict(
        self,
        issues: List[Dict[str, str]],
        threshold: float = 0.30
    ) -> List[Dict]:
        """Predict categories for multiple issues."""
        return [
            self.predict(
                title=issue['title'],
                body=issue['body'],
                repo=issue['repo'],
                threshold=threshold
            )
            for issue in issues
        ]