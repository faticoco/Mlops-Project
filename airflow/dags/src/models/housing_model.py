import os
import logging
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HousingModel:
    """Class for training and evaluating housing price prediction models."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file path."""
        from src.data.housing_processor import HousingDataProcessor
        self.processor = HousingDataProcessor(config_path)
        self.models_path = self.processor.config.get('models_path', 'models')
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
        
        # Set MLflow tracking URI
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        
        # Create experiment or get existing one
        self.experiment_name = "housing_price_prediction"
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new MLflow experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
    
    def load_training_data(self, X_path: str, y_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the preprocessed training data."""
        try:
            X = pd.read_csv(X_path)
            y_df = pd.read_csv(y_path)
            
            # Extract the target column (should be the only column in y_df)
            y = y_df.iloc[:, 0]
            
            logger.info(f"Loaded training data: X shape={X.shape}, y shape={y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str):
        """Load the fitted preprocessor."""
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
            return preprocessor
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise
    
    def prepare_data_for_training(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data for training by handling categorical features and missing values.
        """
        try:
            # Check for missing values before any transformation
            missing_before = X.isnull().sum().sum()
            if missing_before > 0:
                logger.info(f"Found {missing_before} missing values in input data")
            
            # Identify numeric and categorical columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            logger.info(f"Identified {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
            
            # Create a new DataFrame for processed data
            X_processed = X.copy()
            
            # Handle missing values in numeric columns
            if len(numeric_cols) > 0:
                # Fill missing numeric values with median
                for col in numeric_cols:
                    if X_processed[col].isnull().any():
                        median_val = X_processed[col].median()
                        X_processed[col] = X_processed[col].fillna(median_val)
                        logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # Handle missing values in categorical columns
            if len(categorical_cols) > 0:
                logger.info(f"Categorical columns found: {categorical_cols[:5]}... (showing first 5)")
                
                # Fill missing categorical values with most frequent value
                for col in categorical_cols:
                    if X_processed[col].isnull().any():
                        mode_val = X_processed[col].mode()[0]
                        X_processed[col] = X_processed[col].fillna(mode_val)
                        logger.info(f"Filled missing values in {col} with mode: {mode_val}")
                
                # Create a one-hot encoder for categorical columns
                X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
                logger.info(f"Data shape after one-hot encoding: {X_processed.shape}")
            
            # Final check for any remaining missing values
            missing_after = X_processed.isnull().sum().sum()
            if missing_after > 0:
                logger.warning(f"There are still {missing_after} missing values after preprocessing")
                
                # As a last resort, fill any remaining NaN values with 0
                X_processed = X_processed.fillna(0)
                logger.info("Filled any remaining missing values with 0")
            
            return X_processed
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            raise
    
    def create_model_pipelines(self) -> Dict[str, Pipeline]:
        """
        Create scikit-learn pipelines for each model with proper preprocessing steps.
        """
        try:
            # Create a simple imputer for any remaining NaN values
            imputer = SimpleImputer(strategy='mean')
            
            # Define models with preprocessing pipelines
            pipelines = {
                "LinearRegression": Pipeline([
                    ('imputer', imputer),
                    ('model', LinearRegression())
                ]),
                "Ridge": Pipeline([
                    ('imputer', imputer),
                    ('model', Ridge(alpha=1.0))
                ]),
                "Lasso": Pipeline([
                    ('imputer', imputer),
                    ('model', Lasso(alpha=0.1))
                ]),
                "RandomForest": Pipeline([
                    ('imputer', imputer),
                    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
                ]),
                "GradientBoosting": Pipeline([
                    ('imputer', imputer),
                    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ])
            }
            
            return pipelines
        except Exception as e:
            logger.error(f"Error creating model pipelines: {e}")
            raise
    
    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate multiple regression models.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Dictionary with best model and metrics
        """
        try:
            # Prepare data for training (handle categorical features and missing values)
            X_prepared = self.prepare_data_for_training(X)
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_prepared, y, test_size=0.2, random_state=42
            )
            logger.info(f"Split data into train and test sets: X_train={X_train.shape}, X_test={X_test.shape}")
            
            # Create model pipelines
            model_pipelines = self.create_model_pipelines()
            
            # Train and evaluate each model
            best_model = None
            best_model_name = None
            best_r2 = -float('inf')
            all_results = {}
            
            # Track with MLflow
            with mlflow.start_run(experiment_id=self.experiment_id, run_name="model_comparison") as run:
                mlflow.log_param("data_shape", X.shape)
                mlflow.log_param("prepared_data_shape", X_prepared.shape)
                mlflow.log_param("train_test_split_ratio", 0.8)
                
                for model_name, pipeline in model_pipelines.items():
                    logger.info(f"Training {model_name}...")
                    
                    # Train the model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Log metrics
                    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                    
                    # Extract the actual model from the pipeline
                    model = pipeline.named_steps['model']
                    
                    # Store results
                    all_results[model_name] = {
                        "pipeline": pipeline,
                        "model": model,
                        "metrics": {
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae,
                            "r2": r2
                        },
                        "feature_names": X_prepared.columns.tolist()  # Store feature names for later use
                    }
                    
                    # Log to MLflow
                    with mlflow.start_run(experiment_id=self.experiment_id, run_name=model_name, nested=True):
                        # Log parameters
                        mlflow.log_params(model.get_params())
                        
                        # Log metrics
                        mlflow.log_metrics({
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae,
                            "r2": r2
                        })
                        
                        # Log model
                        mlflow.sklearn.log_model(pipeline, f"{model_name}_pipeline")
                    
                    # Update best model if this one is better
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = pipeline
                        best_model_name = model_name
                
                # Log best model name
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_r2", best_r2)
            
            logger.info(f"Best model: {best_model_name} with R² = {best_r2:.4f}")
            
            # Return results
            return {
                "best_model_name": best_model_name,
                "best_model": best_model,
                "best_r2": best_r2,
                "all_results": all_results,
                "feature_names": X_prepared.columns.tolist()  # Include feature names in results
            }
        
        except Exception as e:
            logger.error(f"Error training and evaluating models: {e}")
            raise
    
    def save_model(self, model, model_name: str, feature_names: List[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name of the model
            feature_names: List of feature names used for training
            
        Returns:
            Path to the saved model
        """
        try:
            # Create a timestamp-based filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{timestamp}.pkl"
            model_path = os.path.join(self.models_path, model_filename)
            
            # Create a model package that includes the model and metadata
            model_package = {
                "model": model,
                "model_name": model_name,
                "timestamp": timestamp,
                "feature_names": feature_names
            }
            
            # Save the model package
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            logger.info(f"Saved {model_name} model to {model_path}")
            return model_path
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def end_to_end_training(self, X_path: str, y_path: str, preprocessor_path: str) -> Dict[str, Any]:
        """
        Run the end-to-end model training pipeline.
        
        Args:
            X_path: Path to X features CSV
            y_path: Path to y target CSV
            preprocessor_path: Path to saved preprocessor
            
        Returns:
            Dictionary with training results and model paths
        """
        try:
            # Load the data
            X, y = self.load_training_data(X_path, y_path)
            
            # Load the preprocessor (not needed for prediction since X is already preprocessed,
            # but useful to have for future data processing)
            preprocessor = self.load_preprocessor(preprocessor_path)
            
            # Train and evaluate models
            results = self.train_and_evaluate_models(X, y)
            
            # Save the best model
            best_model_path = self.save_model(
                results["best_model"], 
                results["best_model_name"],
                results.get("feature_names")
            )
            
            # Create report
            report = {
                "best_model_name": results["best_model_name"],
                "best_model_path": best_model_path,
                "best_r2": results["best_r2"],
                "metrics": {
                    model_name: info["metrics"] 
                    for model_name, info in results["all_results"].items()
                },
                "X_shape": X.shape,
                "training_complete": True
            }
            
            logger.info(f"Training pipeline completed successfully")
            return report
        
        except Exception as e:
            logger.error(f"Error in end-to-end training pipeline: {e}")
            raise