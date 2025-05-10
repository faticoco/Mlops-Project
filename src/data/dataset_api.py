# src/data/dataset_api.py
import os
import json
import pandas as pd
import requests
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        
        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def download_kaggle_housing_data(self) -> Dict[str, str]:
        """
        Download the Ames Housing dataset from Kaggle.
        
        Returns:
            Dictionary with paths to train and test datasets
        """
        try:
            # First check if we can use the Kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                
                # Check if the API is properly configured
                api = KaggleApi()
                api.authenticate()
                
                # Define the competition
                competition = 'house-prices-advanced-regression-techniques'
                
                # Path to save the files
                kaggle_dir = os.path.join(self.raw_dir, 'kaggle_housing')
                os.makedirs(kaggle_dir, exist_ok=True)
                
                logger.info(f"Downloading Kaggle Housing dataset using Kaggle API")
                
                # Download files
                api.competition_download_files(competition, path=kaggle_dir)
                
                # Unzip the files
                zip_path = os.path.join(kaggle_dir, f"{competition}.zip")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(kaggle_dir)
                
                # Remove the zip file
                os.remove(zip_path)
                
                train_path = os.path.join(kaggle_dir, 'train.csv')
                test_path = os.path.join(kaggle_dir, 'test.csv')
                
                logger.info(f"Downloaded Kaggle Housing dataset to {kaggle_dir}")
                
                return {
                    'train': train_path,
                    'test': test_path
                }
                
            except (ImportError, Exception) as e:
                logger.warning(f"Kaggle API not available or not configured: {e}")
                logger.info("Falling back to direct download method")
                
                # If Kaggle API fails, use direct links to the dataset (from a reliable GitHub source)
                return self._download_housing_data_from_github()
                
        except Exception as e:
            logger.error(f"Error downloading Kaggle Housing dataset: {e}")
            raise
    
    def _download_housing_data_from_github(self) -> Dict[str, str]:
        """
        Download housing dataset from a reliable GitHub repository.
        
        Returns:
            Dictionary with paths to train and test datasets
        """
        try:
            # Using the provided GitHub links for train.csv and test.csv
            train_url = "https://raw.githubusercontent.com/sjmiller8182/RegressionHousingPrices/master/analysis/data/train.csv"
            test_url = "https://raw.githubusercontent.com/sjmiller8182/RegressionHousingPrices/master/analysis/data/test.csv"
            
            github_dir = os.path.join(self.raw_dir, 'github_housing')
            os.makedirs(github_dir, exist_ok=True)
            
            train_path = os.path.join(github_dir, 'train.csv')
            test_path = os.path.join(github_dir, 'test.csv')
            
            # Download train data if it doesn't exist
            if not os.path.exists(train_path):
                logger.info(f"Downloading housing training data from GitHub")
                response = requests.get(train_url)
                response.raise_for_status()
                
                with open(train_path, 'wb') as f:
                    f.write(response.content)
            
            # Download test data if it doesn't exist
            if not os.path.exists(test_path):
                logger.info(f"Downloading housing test data from GitHub")
                response = requests.get(test_url)
                response.raise_for_status()
                
                with open(test_path, 'wb') as f:
                    f.write(response.content)
            
            logger.info(f"Downloaded housing datasets to {github_dir}")
            
            return {
                'train': train_path,
                'test': test_path
            }
        except Exception as e:
            logger.error(f"Error downloading housing data from GitHub: {e}")
            raise
    
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
            
            # Save to CSV
            output_file = os.path.join(self.raw_dir, 'california_housing.csv')
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
            "kaggle_housing": "Ames Housing dataset from Kaggle - residential home sales in Ames, Iowa",
            "california_housing": "California Housing dataset - housing prices in California districts"
        }