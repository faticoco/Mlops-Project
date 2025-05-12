import os
import json
import pandas as pd
import requests
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetAPI:
    """Class for interacting with open source dataset APIs."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize with data directory path."""
        # The data_dir should already point to the data directory
        # For example, it should be '/opt/airflow/data'
        self.data_dir = data_dir
        
        # Ensure the raw, processed, and features directories exist
        self._ensure_directory_structure()
        
        logger.info(f"Initialized DatasetAPI with data_dir: {self.data_dir}")
        
    def _ensure_directory_structure(self):
        """Ensure all required directories exist with correct permissions."""
        try:
            # Create main data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Created or confirmed data directory exists: {self.data_dir}")
            
            # Create raw directory at top level
            raw_dir = os.path.join(self.data_dir, 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            logger.info(f"Created or confirmed raw directory exists: {raw_dir}")
            
            # Create processed directory at top level
            processed_dir = os.path.join(self.data_dir, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            logger.info(f"Created or confirmed processed directory exists: {processed_dir}")
            
            # Create features directory at top level
            features_dir = os.path.join(self.data_dir, 'features')
            os.makedirs(features_dir, exist_ok=True)
            logger.info(f"Created or confirmed features directory exists: {features_dir}")
            
            # Check permissions for the raw directory
            try:
                subprocess.run(['chmod', '777', raw_dir], check=True)
                logger.info(f"Set full permissions on raw directory: {raw_dir}")
            except Exception as e:
                logger.warning(f"Could not set permissions on raw directory: {e}")
            
            # List the contents to verify
            try:
                result = subprocess.run(['ls', '-la', self.data_dir], capture_output=True, text=True)
                logger.info(f"Data directory structure:\n{result.stdout}")
            except Exception as e:
                logger.warning(f"Could not list directory contents: {e}")
                
            # Check for raw/raw nested structure and clean it up if it exists
            nested_raw = os.path.join(raw_dir, 'raw')
            if os.path.exists(nested_raw):
                try:
                    logger.warning(f"Found nested raw directory: {nested_raw}. Cleaning up...")
                    # Move all files from nested raw to parent raw
                    nested_files = os.listdir(nested_raw)
                    for file in nested_files:
                        src = os.path.join(nested_raw, file)
                        dst = os.path.join(raw_dir, file)
                        if os.path.isfile(src):
                            os.rename(src, dst)
                            logger.info(f"Moved {src} to {dst}")
                    
                    # Remove the nested raw directory
                    if len(os.listdir(nested_raw)) == 0:
                        os.rmdir(nested_raw)
                        logger.info(f"Removed empty nested raw directory: {nested_raw}")
                except Exception as e:
                    logger.warning(f"Error cleaning up nested raw directory: {e}")
            
            logger.info("All required directories have been created")
        except Exception as e:
            logger.error(f"Error creating directory structure: {e}")
            raise
        
    def download_housing_data_from_github(self) -> Dict[str, str]:
        """Download housing data from GitHub."""
        try:
            # The raw directory should be a direct child of the data directory
            raw_dir = os.path.join(self.data_dir, 'raw')
            
            # Make sure it exists
            os.makedirs(raw_dir, exist_ok=True)
            logger.info(f"Using raw directory: {raw_dir}")
            
            # Set the paths for train and test files
            train_path = os.path.join(raw_dir, "train.csv")
            test_path = os.path.join(raw_dir, "test.csv")
            
            # Download train.csv
            train_url = "https://raw.githubusercontent.com/sjmiller8182/RegressionHousingPrices/master/analysis/data/train.csv"
            logger.info(f"Downloading train.csv to {train_path}")
            
            # Use /tmp as a safe temporary location
            temp_train_path = '/tmp/train.csv'
            response = requests.get(train_url)
            with open(temp_train_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded train.csv to {temp_train_path}")
            
            # Copy from temp to the final location
            try:
                subprocess.run(['cp', temp_train_path, train_path], check=True)
                logger.info(f"Copied train.csv to {train_path}")
            except Exception as e:
                logger.error(f"Failed to copy train.csv to {train_path}: {e}")
                # Use the temp file as fallback
                train_path = temp_train_path
                logger.info(f"Using temporary file as fallback: {train_path}")
            
            # Download test.csv
            test_url = "https://raw.githubusercontent.com/sjmiller8182/RegressionHousingPrices/master/analysis/data/test.csv"
            logger.info(f"Downloading test.csv to {test_path}")
            
            # Use /tmp as a safe temporary location
            temp_test_path = '/tmp/test.csv'
            response = requests.get(test_url)
            with open(temp_test_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded test.csv to {temp_test_path}")
            
            # Copy from temp to the final location
            try:
                subprocess.run(['cp', temp_test_path, test_path], check=True)
                logger.info(f"Copied test.csv to {test_path}")
            except Exception as e:
                logger.error(f"Failed to copy test.csv to {test_path}: {e}")
                # Use the temp file as fallback
                test_path = temp_test_path
                logger.info(f"Using temporary file as fallback: {test_path}")
            
            # Check file existence to confirm
            if os.path.exists(train_path):
                logger.info(f"Confirmed train.csv exists at {train_path}")
            else:
                logger.error(f"train.csv not found at {train_path}")
                
            if os.path.exists(test_path):
                logger.info(f"Confirmed test.csv exists at {test_path}")
            else:
                logger.error(f"test.csv not found at {test_path}")
            
            logger.info(f"Housing datasets available at: train={train_path}, test={test_path}")
            return {"train": train_path, "test": test_path}
        except Exception as e:
            logger.error(f"Error downloading housing data: {e}")
            
            # Try a fallback approach using /tmp directory
            logger.info("Trying fallback approach with /tmp directory")
            train_path = "/tmp/train.csv"
            test_path = "/tmp/test.csv"
            
            try:
                # Download train.csv directly to /tmp
                train_url = "https://raw.githubusercontent.com/sjmiller8182/RegressionHousingPrices/master/analysis/data/train.csv"
                response = requests.get(train_url)
                with open(train_path, "wb") as f:
                    f.write(response.content)
                
                # Download test.csv directly to /tmp
                test_url = "https://raw.githubusercontent.com/sjmiller8182/RegressionHousingPrices/master/analysis/data/test.csv"
                response = requests.get(test_url)
                with open(test_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Successfully saved files to temporary location: train={train_path}, test={test_path}")
                return {"train": train_path, "test": test_path}
            except Exception as fallback_e:
                logger.error(f"Fallback approach also failed: {fallback_e}")
                raise fallback_e
        
    def fetch_california_housing_from_sklearn(self) -> str:
        """
        Fetch the California Housing dataset from scikit-learn and save it locally.
        Returns the path to the saved dataset.
        """
        try:
            from sklearn.datasets import fetch_california_housing
            
            # California Housing data
            california = fetch_california_housing()
            
            # Create DataFrame
            df = pd.DataFrame(
                data=california.data,
                columns=california.feature_names
            )
            # Add target column
            df['PRICE'] = california.target
            
            # Use the raw directory
            raw_dir = os.path.join(self.data_dir, 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            
            # Save to the raw directory
            output_file = os.path.join(raw_dir, 'california_housing.csv')
            df.to_csv(output_file, index=False)
            
            logger.info(f"California Housing dataset saved to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error fetching California Housing dataset: {e}")
            raise
            
    def list_available_datasets(self) -> Dict[str, str]:
        """
        Return a dictionary of datasets that can be downloaded using this API.
        """
        return {
            "github_housing": "Ames Housing dataset from GitHub - residential home sales in Ames, Iowa",
            "california_housing": "California Housing dataset - housing prices in California districts"
        }