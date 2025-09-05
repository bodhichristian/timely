"""
Utility functions for model persistence and loading.
"""
import os
import joblib
from sklearn.calibration import CalibratedClassifierCV

def save_model_artifacts(model, tfidf_vectorizer, label_encoder, repo_encoder, output_dir):
    """
    Save all model artifacts needed for prediction
    
    Args:
        model: Trained XGBoost model
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Fitted LabelEncoder for categories
        repo_encoder: Fitted LabelEncoder for repositories
        output_dir: Directory to save artifacts
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    artifacts = {
        'model': model,
        'tfidf_vectorizer': tfidf_vectorizer,
        'label_encoder': label_encoder,
        'repo_encoder': repo_encoder
    }
    
    for name, artifact in artifacts.items():
        joblib.dump(artifact, os.path.join(output_dir, f'{name}.joblib'))

def load_model_artifacts(model_dir):
    """
    Load all model artifacts needed for prediction
    
    Args:
        model_dir: Directory containing saved artifacts
        
    Returns:
        dict: Loaded artifacts
    """
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist")
        
    artifacts = {}
    required_files = ['model', 'tfidf_vectorizer', 'label_encoder', 'repo_encoder']
    
    for name in required_files:
        file_path = os.path.join(model_dir, f'{name}.joblib')
        if not os.path.exists(file_path):
            raise ValueError(f"Required artifact {file_path} not found")
        artifacts[name] = joblib.load(file_path)
    
    return artifacts

def calibrate_xgboost_model(xgb_model, X_val, y_val):
    """
    Calibrate XGBoost probability predictions
    
    Args:
        xgb_model: Trained XGBoost model
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Calibrated model
    """
    calibrated_model = CalibratedClassifierCV(
        xgb_model, 
        cv='prefit',
        method='sigmoid'
    )
    calibrated_model.fit(X_val, y_val)
    return calibrated_model
