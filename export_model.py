"""
Export trained model and artifacts for production use.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from model_utils import save_model_artifacts

# Load the processed data
df = pd.read_csv('github_issues_processed.csv')

# Initialize and fit TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=250, stop_words='english', ngram_range=(1,2))
combined_text = df[['title', 'body']].fillna('').apply(lambda x: ' '.join(x), axis=1)
tfidf.fit(combined_text)

# Initialize and fit label encoder for categories
le = LabelEncoder()
le.fit(df['category'])

# Initialize and fit repo encoder
repo_encoder = LabelEncoder()
repo_encoder.fit(df['repo_name'])

# Configure XGBoost
xgb_config = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'multi:softprob',
    'num_class': len(le.classes_)
}

# Create and train XGBoost model
model = xgb.XGBClassifier(**xgb_config)
model.fit(X_train, y_cat_train_encoded)  # Using variables from notebook

# Save artifacts
save_model_artifacts(
    model=model,
    tfidf_vectorizer=tfidf,
    label_encoder=le,
    repo_encoder=repo_encoder,
    output_dir='model_artifacts'
)
