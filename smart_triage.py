"""
Smart issue triage system using XGBoost for category prediction.
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
        
        # Define confidence thresholds based on model performance
        self.confidence_thresholds = {
            'high': 0.8,    # High confidence threshold
            'medium': 0.6,  # Medium confidence threshold
            'low': 0.4      # Low confidence threshold
        }
    
    def get_recommendations(
        self,
        features: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict:
        """
        Generate recommendations based on model predictions
        
        Args:
            features: Processed features
            threshold: Confidence threshold for secondary suggestions
            
        Returns:
            dict: Recommendations and insights
        """
        # Get model predictions and probabilities
        proba = self.model.predict_proba(features)[0]
        
        # Convert encoded predictions back to original labels
        classes = self.label_encoder.inverse_transform(range(len(proba)))
        
        # Sort predictions by confidence
        pred_confidence = list(zip(classes, proba))
        pred_confidence.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare recommendations
        recommendations = {
            'primary_category': {
                'category': pred_confidence[0][0],
                'confidence': float(pred_confidence[0][1]),
                'action_needed': True if pred_confidence[0][0] in ['is_bug_cat', 'is_priority_cat'] else False
            },
            'secondary_suggestions': [
                {
                    'category': cat,
                    'confidence': float(conf),
                    'action_needed': True if cat in ['is_bug_cat', 'is_priority_cat'] else False
                }
                for cat, conf in pred_confidence[1:3] if conf > threshold
            ],
            'triage_recommendations': []
        }
        
        # Add triage recommendations based on predictions
        primary_conf = recommendations['primary_category']['confidence']
        primary_cat = recommendations['primary_category']['category']
        
        if primary_cat == 'is_bug_cat':
            if primary_conf > self.confidence_thresholds['high']:
                recommendations['triage_recommendations'].append({
                    'type': 'high_confidence_bug',
                    'message': 'High confidence bug report - Immediate review recommended',
                    'priority': 'high'
                })
            elif primary_conf > self.confidence_thresholds['medium']:
                recommendations['triage_recommendations'].append({
                    'type': 'medium_confidence_bug',
                    'message': 'Medium confidence bug report - Review within 24 hours',
                    'priority': 'medium'
                })
        elif primary_cat == 'is_feature_cat':
            if primary_conf > self.confidence_thresholds['medium']:
                recommendations['triage_recommendations'].append({
                    'type': 'clear_feature_request',
                    'message': 'Clear feature request - Add to product backlog',
                    'priority': 'medium'
                })
        elif primary_cat == 'is_doc_cat':
            if primary_conf > self.confidence_thresholds['medium']:
                recommendations['triage_recommendations'].append({
                    'type': 'documentation',
                    'message': 'Documentation issue - Tag for docs team review',
                    'priority': 'medium'
                })
        
        # Add confidence-based recommendations
        if primary_conf < self.confidence_thresholds['low']:
            recommendations['triage_recommendations'].append({
                'type': 'low_confidence',
                'message': 'Low confidence prediction - Manual review recommended',
                'priority': 'medium'
            })
            
            # Add secondary suggestions with lower threshold
            low_conf_suggestions = [
                {
                    'category': cat,
                    'confidence': float(conf),
                    'action_needed': True if cat in ['is_bug_cat', 'is_priority_cat'] else False
                }
                for cat, conf in pred_confidence[1:4] if conf > threshold / 2
            ]
            recommendations['secondary_suggestions'].extend(low_conf_suggestions)
        
        return recommendations
    
    def predict(
        self,
        title: str,
        body: str,
        repo: str,
        threshold: float = 0.2
    ) -> Dict:
        """
        Predict issue categories and provide recommendations
        
        Args:
            title: Issue title
            body: Issue description
            repo: Repository name
            threshold: Confidence threshold for secondary suggestions
            
        Returns:
            dict: Predictions and recommendations
        """
        # Combine text
        text = f"{title}\n{body}"
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            text=text,
            repo=repo,
            repo_encoder=self.repo_encoder
        )
        
        # Get recommendations
        recommendations = self.get_recommendations(
            features=features,
            threshold=threshold
        )
        
        # Add repository context
        recommendations['repo_context'] = {
            'repository': repo,
            'typical_response_time': '2-3 days',  # This could be calculated from historical data
            'similar_issues_count': 5  # This could be calculated using embedding similarity
        }
        
        return recommendations
    
    def batch_predict(
        self,
        issues: List[Dict[str, str]],
        threshold: float = 0.2
    ) -> List[Dict]:
        """
        Predict categories for multiple issues
        
        Args:
            issues: List of dicts containing title, body, and repo
            threshold: Confidence threshold for secondary suggestions
            
        Returns:
            list: Predictions for each issue
        """
        return [
            self.predict(
                title=issue['title'],
                body=issue['body'],
                repo=issue['repo'],
                threshold=threshold
            )
            for issue in issues
        ]