# src/models/train_housing_model.py
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ML imports
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def load_data(X_path: str, y_path: str, preprocessor_path: str = None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Load and preprocess the data.
    
    Args:
        X_path: Path to features CSV
        y_path: Path to target CSV
        preprocessor_path: Path to saved preprocessor (optional)
        
    Returns:
        X: Features array
        y: Target array
        preprocessor: Loaded preprocessor (if provided)
    """
    try:
        # Load feature data
        X = pd.read_csv(X_path)
        
        # Load target data
        y_df = pd.read_csv(y_path)
        y = y_df.iloc[:, 0].values  # Take the first column regardless of name
        
        # Load preprocessor if provided
        preprocessor = None
        if preprocessor_path and os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
                
            # Apply preprocessing
            X = preprocessor.transform(X)
        
        logger.info(f"Loaded data - X shape: {X.shape}, y shape: {y.shape}")
        return X, y, preprocessor
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_models() -> Dict[str, Any]:
    """Create various regression models for evaluation."""
    models = {
        "ridge": Ridge(alpha=0.5, random_state=42),
        "lasso": Lasso(alpha=0.001, random_state=42),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
        "svr": SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    }
    
    # Create a stacked model
    estimators = [
        ('ridge', models['ridge']),
        ('lasso', models['lasso']),
        ('svr', models['svr']),
        ('rf', models['random_forest'])
    ]
    
    models["stacked_model"] = StackingRegressor(
        estimators=estimators,
        final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42)
    )
    
    return models

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str, log_transformed: bool = False) -> Dict[str, float]:
    """
    Train and evaluate a model.
    
    Args:
        model: The model to evaluate
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        log_transformed: Whether the target is log-transformed
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # If the target was log-transformed, calculate metrics on original scale as well
    if log_transformed:
        y_train_exp = np.expm1(y_train)
        y_test_exp = np.expm1(y_test)
        y_pred_train_exp = np.expm1(y_pred_train)
        y_pred_test_exp = np.expm1(y_pred_test)
        
        rmse_train_original = np.sqrt(mean_squared_error(y_train_exp, y_pred_train_exp))
        rmse_test_original = np.sqrt(mean_squared_error(y_test_exp, y_pred_test_exp))
        mae_test_original = mean_absolute_error(y_test_exp, y_pred_test_exp)
        r2_test_original = r2_score(y_test_exp, y_pred_test_exp)
        
        metrics = {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test,
            "rmse_train_original": rmse_train_original,
            "rmse_test_original": rmse_test_original,
            "mae_test_original": mae_test_original,
            "r2_test_original": r2_test_original
        }
    else:
        metrics = {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test
        }
    
    # Log metrics
    logger.info(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics

def run_cross_validation(model, X, y, model_name: str, n_splits: int = 5) -> Dict[str, float]:
    """
    Run cross-validation for a model.
    
    Args:
        model: The model to evaluate
        X: Features
        y: Target
        model_name: Name of the model
        n_splits: Number of CV splits
        
    Returns:
        Dictionary of CV metrics
    """
    # Define CV strategy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Run CV for RMSE
    cv_rmse = -cross_val_score(
        model, X, y, 
        scoring='neg_root_mean_squared_error',
        cv=kf, 
        n_jobs=-1
    )
    
    # Run CV for R²
    cv_r2 = cross_val_score(
        model, X, y, 
        scoring='r2',
        cv=kf, 
        n_jobs=-1
    )
    
    # Calculate mean and std
    rmse_mean = cv_rmse.mean()
    rmse_std = cv_rmse.std()
    r2_mean = cv_r2.mean()
    r2_std = cv_r2.std()
    
    cv_metrics = {
        "cv_rmse_mean": rmse_mean,
        "cv_rmse_std": rmse_std,
        "cv_r2_mean": r2_mean,
        "cv_r2_std": r2_std
    }
    
    # Log CV metrics
    logger.info(f"Cross-validation for {model_name}:")
    logger.info(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    logger.info(f"  R²: {r2_mean:.4f} ± {r2_std:.4f}")
    
    return cv_metrics

def save_model(model, model_name: str, output_dir: str) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        output_dir: Directory to save the model
        
    Returns:
        Path to the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to {model_path}")
    return model_path

def log_to_mlflow(model, model_name: str, params: Dict[str, Any], metrics: Dict[str, float], 
                  cv_metrics: Dict[str, float], feature_names: List[str], experiment_name: str) -> str:
    """
    Log model training results to MLflow.
    
    Args:
        model: Trained model
        model_name: Name of the model
        params: Model parameters
        metrics: Evaluation metrics
        cv_metrics: Cross-validation metrics
        feature_names: Names of the features
        experiment_name: MLflow experiment name
        
    Returns:
        MLflow run ID
    """
    # Set up MLflow tracking
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name) as run:
        # Log model parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log cross-validation metrics
        for metric_name, metric_value in cv_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model type
        mlflow.log_param("model_type", model_name)
        
        # Log feature names
        mlflow.log_param("feature_count", len(feature_names))
        
        # Log the model
        mlflow.sklearn.log_model(model, f"models/{model_name}")
        
        run_id = run.info.run_id
        logger.info(f"Logged results to MLflow, run_id: {run_id}")
        
        return run_id

def plot_feature_importance(model, feature_names: List[str], output_dir: str, model_name: str):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names: Names of the features
        output_dir: Directory to save the plot
        model_name: Name of the model
    """
    # Only plot for models that have feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Take top 20 features
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances, align='center')
    plt.yticks(range(top_n), top_feature_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances for {model_name}')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Saved feature importance plot to {plot_path}")

def main():
    """Main function to train and evaluate models."""
    parser = argparse.ArgumentParser(description='Train housing price prediction models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up MLflow
    mlflow_tracking_uri = config.get('mlflow_tracking_uri', os.path.join(args.output_dir, 'mlruns'))
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = config.get('experiment_name', 'housing_price_prediction')
    
    # Determine dataset to use
    dataset_type = config.get('dataset_type', 'kaggle_housing')
    
    # Get the latest data files
    data_dir = args.data_dir
    features_dir = os.path.join(data_dir, 'features')
    
    # Find the latest X and y files for the selected dataset
    x_files = sorted([f for f in os.listdir(features_dir) if f.startswith(f'X_{dataset_type}')])
    y_files = sorted([f for f in os.listdir(features_dir) if f.startswith(f'y_{dataset_type}')])
    
    if not x_files or not y_files:
        logger.error(f"No data files found for {dataset_type}. Please run data processing pipeline first.")
        return
    
    latest_x = os.path.join(features_dir, x_files[-1])
    latest_y = os.path.join(features_dir, y_files[-1])
    
    # Find the matching preprocessor
    preprocessor_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'processed')) 
                                if f.startswith(f'preprocessor_{dataset_type}')])
    
    preprocessor_path = None
    if preprocessor_files:
        preprocessor_path = os.path.join(data_dir, 'processed', preprocessor_files[-1])
    
    # Load data
    X, y, preprocessor = load_data(latest_x, latest_y, preprocessor_path)
    
    # Get feature names (from X dataframe or preprocessor)
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    elif preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if target is log-transformed
    log_transformed = config.get('log_transform_target', False)
    
    # Create models
    models = create_models()
    
    # Evaluate models
    model_metrics = {}
    cv_metrics = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} model")
        
        # Run cross-validation
        model_cv_metrics = run_cross_validation(model, X_train, y_train, model_name)
        cv_metrics[model_name] = model_cv_metrics
        
        # Train and evaluate
        model_eval_metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name, log_transformed)
        model_metrics[model_name] = model_eval_metrics
        
        # Save model
        model_dir = os.path.join(args.output_dir, 'models')
        save_model(model, model_name, model_dir)
        
        # Plot feature importance (for applicable models)
        plots_dir = os.path.join(args.output_dir, 'plots')
        plot_feature_importance(model, feature_names, plots_dir, model_name)
        
        # Get model parameters
        params = {}
        for param_name in model.get_params():
            params[param_name] = str(model.get_params()[param_name])
        
        # Log to MLflow
        log_to_mlflow(model, model_name, params, model_eval_metrics, model_cv_metrics, 
                      feature_names, experiment_name)
    
    # Find best model based on test RMSE
    best_model_name = min(model_metrics.items(), key=lambda x: x[1]['rmse_test'])[0]
    
    # Log best model
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Test RMSE: {model_metrics[best_model_name]['rmse_test']:.4f}")
    
    # Save best model separately
    best_model = models[best_model_name]
    best_model_path = os.path.join(args.output_dir, 'models', 'best_model.pkl')
    
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    logger.info(f"Saved best model to {best_model_path}")
    
    # Create a model comparison report
    report = pd.DataFrame({
        'Model': list(model_metrics.keys()),
        'Test RMSE': [metrics['rmse_test'] for metrics in model_metrics.values()],
        'Test R²': [metrics['r2_test'] for metrics in model_metrics.values()],
        'CV RMSE': [metrics['cv_rmse_mean'] for metrics in cv_metrics.values()],
        'CV R²': [metrics['cv_r2_mean'] for metrics in cv_metrics.values()]
    })
    
    # Sort by Test RMSE
    report = report.sort_values('Test RMSE')
    
    # Save report
    report_path = os.path.join(args.output_dir, 'model_comparison.csv')
    report.to_csv(report_path, index=False)
    
    logger.info(f"Saved model comparison report to {report_path}")
    
    # Print report
    print("\nModel Comparison Report:")
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()