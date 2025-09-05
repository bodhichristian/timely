# Smart Issue Triage

A machine learning-powered system for automatically categorizing and triaging GitHub issues. The system uses XGBoost with TF-IDF and BERT embeddings to provide accurate category predictions with confidence scores.

## Features

- Automatic issue categorization (bug, feature request, documentation, etc.)
- Confidence scores for predictions
- Multiple category suggestions when appropriate
- Smart triage recommendations based on issue content
- Support for batch predictions
- Production-ready with model persistence

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd timely
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from smart_triage import SmartIssueTriage

# Initialize the system
triage = SmartIssueTriage(model_dir='path/to/model/artifacts')

# Make a prediction
result = triage.predict(
    title="Error in login flow",
    body="Users cannot reset their password",
    repo="auth-service"
)

# Print primary category and confidence
print(f"Category: {result['primary_category']['category']}")
print(f"Confidence: {result['primary_category']['confidence']:.2%}")
```

### Batch Predictions

```python
issues = [
    {
        'title': 'Error in login flow',
        'body': 'Users cannot reset their password',
        'repo': 'auth-service'
    },
    {
        'title': 'Add dark mode support',
        'body': 'Implement dark mode theme for better accessibility',
        'repo': 'frontend-ui'
    }
]

results = triage.batch_predict(issues)
```

## Model Performance

The current XGBoost model achieves:
- Accuracy: 0.94
- Precision: 0.93
- Recall: 0.94
- F1 Score: 0.94
- Average Confidence: 0.95

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.