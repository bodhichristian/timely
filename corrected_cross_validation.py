# Corrected Cross-Validation Code
# Copy this into your notebook cell

# Define scoring metrics
scoring_metrics = {
    'mae': 'neg_mean_absolute_error',
    'mse': 'neg_mean_squared_error',
    'r2': 'r2'
}

# Initialize results dictionary
results = {}

n_jobs = 8

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=n_jobs
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42
    ),
}

# Use same core count for CV
for name, model in models.items():
    print(f"Training {name} using {n_jobs} cores...")
    model_results = {}
    for metric_name, metric in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric, n_jobs=n_jobs)
        if metric_name in ['mae', 'mse']:
            model_results[metric_name] = -scores.mean()
        else:
            model_results[metric_name] = scores.mean()
        model_results[f'{metric_name}_std'] = scores.std()
    results[name] = model_results

# Display results
results_df = pd.DataFrame(results).T
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("="*80)
print(results_df.round(4))
