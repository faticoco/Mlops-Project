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
        
    def download_housing_data_from_github(self) -> Dict[str, str]:
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
            "github_housing": "Ames Housing dataset from GitHub - residential home sales in Ames, Iowa",
            "california_housing": "California Housing dataset - housing prices in California districts"
        }