"""
Test script for Smart Issue Triage system.
"""
from smart_triage import SmartIssueTriage
import json

def print_recommendations(result):
    """Pretty print the triage recommendations"""
    print("\nPrimary Category:")
    print(f"  • Category: {result['primary_category']['category']}")
    print(f"  • Confidence: {result['primary_category']['confidence']:.2%}")
    print(f"  • Action Needed: {result['primary_category']['action_needed']}")
    
    if result['secondary_suggestions']:
        print("\nSecondary Suggestions:")
        for suggestion in result['secondary_suggestions']:
            print(f"  • {suggestion['category']} (Confidence: {suggestion['confidence']:.2%})")
    
    if result['triage_recommendations']:
        print("\nTriage Recommendations:")
        for rec in result['triage_recommendations']:
            print(f"  • [{rec['priority'].upper()}] {rec['message']}")
    
    if 'repo_context' in result:
        print("\nRepository Context:")
        print(f"  • Repository: {result['repo_context']['repository']}")
        print(f"  • Typical Response Time: {result['repo_context']['typical_response_time']}")
        print(f"  • Similar Issues: {result['repo_context']['similar_issues_count']}")

def main():
    # Initialize the triage system
    triage = SmartIssueTriage(model_dir='model_artifacts')
    
    # Test cases
    test_issues = [
        {
            'title': 'Cannot login after password reset',
            'body': '''After requesting a password reset and setting a new password,
                      the login form keeps showing "Invalid credentials".
                      Steps to reproduce:
                      1. Click "Forgot Password"
                      2. Receive and use reset link
                      3. Set new password
                      4. Try to login
                      
                      Error occurs consistently across Chrome and Firefox.''',
            'repo': 'angular/angular'
        },
        {
            'title': 'Add dark mode support to dashboard',
            'body': '''Request to implement dark mode theme for the dashboard.
                      This would improve accessibility and reduce eye strain for users
                      working at night.
                      
                      Suggested implementation:
                      - Add theme toggle in user settings
                      - Use CSS variables for color scheme
                      - Support system preference detection''',
            'repo': 'microsoft/vscode'
        },
        {
            'title': 'Update API documentation for new endpoints',
            'body': '''The documentation for the following new endpoints is missing:
                      - POST /api/v2/users/verify
                      - GET /api/v2/users/settings
                      
                      Please add:
                      - Request/response formats
                      - Authentication requirements
                      - Example usage''',
            'repo': 'angular/angular'
        }
    ]
    
    print("Smart Issue Triage Test\n" + "="*50)
    
    # Test individual prediction
    print("\nTesting Single Prediction:")
    print("-"*30)
    result = triage.predict(
        title=test_issues[0]['title'],
        body=test_issues[0]['body'],
        repo=test_issues[0]['repo']
    )
    print_recommendations(result)
    
    # Test batch prediction
    print("\n\nTesting Batch Prediction:")
    print("-"*30)
    results = triage.batch_predict(test_issues)
    
    for i, result in enumerate(results, 1):
        print(f"\nIssue {i}:")
        print("-"*10)
        print(f"Title: {test_issues[i-1]['title']}")
        print_recommendations(result)
        print("\n" + "="*50)
    
    # Save results to file
    with open('triage_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to triage_test_results.json")

if __name__ == "__main__":
    main()