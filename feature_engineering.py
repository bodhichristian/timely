"""
Feature engineering utilities for text processing and embeddings.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer

class TextFeatureExtractor:
    def __init__(self, tfidf_vectorizer, bert_model_name='all-MiniLM-L6-v2'):
        """
        Initialize text feature extractor
        
        Args:
            tfidf_vectorizer: Pre-trained TF-IDF vectorizer
            bert_model_name: Name of the BERT model to use
        """
        self.tfidf_vectorizer = tfidf_vectorizer
        self.bert_model = SentenceTransformer(bert_model_name)
        
        # Define the exact order of features expected by the trained model
        # IMPORTANT: This must match the DataFrame columns used during training exactly
        self.feature_order = [
            'created_hour',
            'created_day_of_week',
            'created_month',
            'n_days_to_resolution',
            'title_length',
            'body_length',
            'title_word_count',
            'body_word_count',
            'code_block_count',
            'url_count',
            'title_question_word_count',
            'title_has_question_mark',
            'body_question_word_count',
            'body_has_question_mark',
            'total_question_word_count',
            'total_has_question_mark',
            'includes_questions',
            'title_n_urgent_words',
            'title_has_exclamation',
            'body_n_urgent_words',
            'body_has_exclamation',
            'total_n_urgent_words',
            'total_has_exclamation',
            'urgency_score',
            'repo_encoded'
        ]
        
        # Add TF-IDF feature names (250 features as in training)
        self.feature_order.extend([f'tfidf_{i}' for i in range(250)])
        
        # Add BERT feature names (384 features as in training)
        self.feature_order.extend([f'bert_{i}' for i in range(384)])
        
    def extract_basic_features(self, text, title, body):
        """
        Extract basic text features
        
        Args:
            text: Combined text
            title: Issue title
            body: Issue body
            
        Returns:
            dict: Basic text features
        """
        features = {}
        
        # Initialize all features to 0
        for feature in self.feature_order:
            features[feature] = 0
        
        # Title features
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        features['title_has_question_mark'] = int('?' in title)
        features['title_has_exclamation'] = int('!' in title)
        
        # Body features
        features['body_length'] = len(body)
        features['body_word_count'] = len(body.split())
        features['body_has_question_mark'] = int('?' in body)
        features['body_has_exclamation'] = int('!' in body)
        
        # Code block features
        features['code_block_count'] = body.count('```') // 2
        
        # URL features
        features['url_count'] = body.lower().count('http')
        
        # Question indicators
        question_words = ['how', 'what', 'why', 'when', 'where', 'which', 'who']
        features['title_question_word_count'] = sum(title.lower().count(word) for word in question_words)
        features['body_question_word_count'] = sum(body.lower().count(word) for word in question_words)
        features['total_question_word_count'] = features['title_question_word_count'] + features['body_question_word_count']
        features['total_has_question_mark'] = features['title_has_question_mark'] + features['body_has_question_mark']
        features['includes_questions'] = int((features['total_question_word_count'] > 0) or (features['total_has_question_mark'] > 0))
        
        # Urgency indicators
        urgent_words = ['urgent', 'critical', 'asap', 'immediate', 'emergency', 
                       'broken', 'error', 'serious', 'security']
        features['title_n_urgent_words'] = sum(title.lower().count(word) for word in urgent_words)
        features['body_n_urgent_words'] = sum(body.lower().count(word) for word in urgent_words)
        features['total_n_urgent_words'] = features['title_n_urgent_words'] + features['body_n_urgent_words']
        features['total_has_exclamation'] = features['title_has_exclamation'] + features['body_has_exclamation']
        features['urgency_score'] = features['total_n_urgent_words'] + features['total_has_exclamation']
        
        return features
    
    def extract_tfidf_features(self, text):
        """
        Extract TF-IDF features
        
        Args:
            text: Input text
            
        Returns:
            dict: TF-IDF features
        """
        tfidf_features = self.tfidf_vectorizer.transform([text])
        features = {}
        for i in range(250):
            feature_name = f'tfidf_{i}'
            features[feature_name] = tfidf_features[0, i] if i < tfidf_features.shape[1] else 0
        return features
    
    def extract_bert_features(self, text):
        """
        Extract BERT embeddings
        
        Args:
            text: Input text
            
        Returns:
            dict: BERT features
        """
        bert_embedding = self.bert_model.encode([text])
        features = {}
        # Support both (1, 384) and (384,) shapes
        if hasattr(bert_embedding, 'shape') and len(bert_embedding.shape) == 2:
            for i in range(384):
                feature_name = f'bert_{i}'
                features[feature_name] = bert_embedding[0, i] if i < bert_embedding.shape[1] else 0
        else:
            for i in range(384):
                feature_name = f'bert_{i}'
                features[feature_name] = bert_embedding[i] if i < len(bert_embedding) else 0
        return features
    
    def extract_all_features(self, text, repo, repo_encoder):
        """
        Extract all features for a given text
        
        Args:
            text: Input text
            repo: Repository name
            repo_encoder: Fitted LabelEncoder for repositories
            
        Returns:
            pd.DataFrame: All extracted features
        """
        # Split text into title and body
        parts = text.split('\n', 1)
        title = parts[0]
        body = parts[1] if len(parts) > 1 else ''
        
        # Extract all features
        features = {}
        
        # Get basic features
        features.update(self.extract_basic_features(text, title, body))
        
        # Add repository encoding
        features['repo_encoded'] = repo_encoder.transform([repo])[0]
        
        # Get TF-IDF features
        features.update(self.extract_tfidf_features(text))
        
        # Get BERT features
        features.update(self.extract_bert_features(text))
        
        # Create DataFrame with exact feature order
        df = pd.DataFrame([features])[self.feature_order]
        
        return df