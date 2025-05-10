
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import yaml
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HousingDataProcessor:
    """Class for processing housing datasets."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file path."""
        self.config = self._load_config(config_path)
        self.raw_data_path = self.config['raw_data_path']
        self.processed_data_path = self.config['processed_data_path']
        self.features_path = self.config['features_path']
        
        # Ensure directories exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load the housing dataset."""
        try:
            filepath = os.path.join(self.raw_data_path, filename)
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_ames_housing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the Ames Housing dataset.
        
        Args:
            df: Raw Ames Housing dataset
            
        Returns:
            Cleaned dataframe
        """
        try:
            # Make a copy to avoid modifying the original
            df_cleaned = df.copy()
            
            # Handle missing values
            # For Ames Housing dataset, missing values are often encoded in different ways
            
            # Some columns use 'NA' to represent "Not Applicable" rather than missing
            na_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                      'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                      'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
            
            # Replace 'NA' with None in these columns
            for col in na_cols:
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].replace('NA', np.nan)
            
            # Drop columns with too many missing values (e.g., more than 80%)
            missing_percentages = df_cleaned.isnull().mean()
            cols_to_drop = missing_percentages[missing_percentages > 0.8].index.tolist()
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >80% missing values")
            
            # Drop rows with too many missing values (e.g., more than 20%)
            df_cleaned = df_cleaned.dropna(thresh=df_cleaned.shape[1] * 0.8)
            logger.info(f"Remaining rows after dropping rows with >20% missing values: {df_cleaned.shape[0]}")
            
            # Convert specific columns to appropriate types
            # Year columns to categorical
            year_cols = [col for col in df_cleaned.columns if 'Yr' in col or 'Year' in col]
            for col in year_cols:
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].astype('str')
            
            # Remove outliers in SalePrice (if present)
            if 'SalePrice' in df_cleaned.columns:
                q1 = df_cleaned['SalePrice'].quantile(0.25)
                q3 = df_cleaned['SalePrice'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                
                df_cleaned = df_cleaned[(df_cleaned['SalePrice'] >= lower_bound) & 
                                        (df_cleaned['SalePrice'] <= upper_bound)]
                logger.info(f"Removed SalePrice outliers. Remaining rows: {df_cleaned.shape[0]}")
            
            # Remove duplicate rows
            df_cleaned = df_cleaned.drop_duplicates()
            logger.info(f"Final shape after cleaning: {df_cleaned.shape}")
            
            return df_cleaned
        except Exception as e:
            logger.error(f"Error cleaning Ames Housing data: {e}")
            raise
    
    def clean_california_housing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the California Housing dataset.
        
        Args:
            df: Raw California Housing dataset
            
        Returns:
            Cleaned dataframe
        """
        try:
            # Make a copy to avoid modifying the original
            df_cleaned = df.copy()
            
            # Handle missing values (if any)
            missing_percentages = df_cleaned.isnull().mean()
            cols_to_drop = missing_percentages[missing_percentages > 0.2].index.tolist()
            if cols_to_drop:
                df_cleaned = df_cleaned.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} columns with >20% missing values")
            
            # Remove duplicate rows
            df_cleaned = df_cleaned.drop_duplicates()
            
            # Remove outliers in target variable (PRICE)
            if 'PRICE' in df_cleaned.columns:
                q1 = df_cleaned['PRICE'].quantile(0.25)
                q3 = df_cleaned['PRICE'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                
                df_cleaned = df_cleaned[(df_cleaned['PRICE'] >= lower_bound) & 
                                        (df_cleaned['PRICE'] <= upper_bound)]
                logger.info(f"Removed PRICE outliers. Remaining rows: {df_cleaned.shape[0]}")
            
            logger.info(f"Final shape after cleaning: {df_cleaned.shape}")
            return df_cleaned
        except Exception as e:
            logger.error(f"Error cleaning California Housing data: {e}")
            raise
    
    def create_features_ames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from the Ames Housing dataset.
        
        Args:
            df: Cleaned Ames Housing dataset
            
        Returns:
            Dataframe with engineered features
        """
        try:
            # Make a copy to avoid modifying the original
            df_features = df.copy()
            
            # Create new features
            
            # 1. Total SF = Total square footage from basement, 1st and 2nd floors
            if all(col in df_features.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
                df_features['TotalSF'] = df_features['TotalBsmtSF'] + df_features['1stFlrSF'] + df_features['2ndFlrSF']
            
            # 2. House Age (if YearBuilt exists)
            if 'YearBuilt' in df_features.columns and df_features['YearBuilt'].dtype == 'str':
                # Convert back to numeric for calculation
                df_features['YearBuilt'] = pd.to_numeric(df_features['YearBuilt'], errors='coerce')
                # Calculate age
                current_year = 2025  # Assuming current year
                df_features['HouseAge'] = current_year - df_features['YearBuilt']
                # Convert back to string as we did earlier
                df_features['YearBuilt'] = df_features['YearBuilt'].astype('str')
            
            # 3. Total Bathrooms
            bath_cols = [col for col in df_features.columns if 'Bath' in col]
            if bath_cols:
                df_features['TotalBaths'] = df_features[bath_cols].sum(axis=1)
            
            # 4. Has Pool (binary feature)
            if 'PoolArea' in df_features.columns:
                df_features['HasPool'] = (df_features['PoolArea'] > 0).astype(int)
            
            # 5. Has Garage (binary feature)
            if 'GarageArea' in df_features.columns:
                df_features['HasGarage'] = (df_features['GarageArea'] > 0).astype(int)
            
            # 6. Remodeled (binary feature)
            if all(col in df_features.columns for col in ['YearBuilt', 'YearRemodAdd']):
                # Convert to numeric for comparison
                df_features['YearBuilt'] = pd.to_numeric(df_features['YearBuilt'], errors='coerce')
                df_features['YearRemodAdd'] = pd.to_numeric(df_features['YearRemodAdd'], errors='coerce')
                
                df_features['Remodeled'] = (df_features['YearRemodAdd'] > df_features['YearBuilt']).astype(int)
                
                # Convert back to string
                df_features['YearBuilt'] = df_features['YearBuilt'].astype('str')
                df_features['YearRemodAdd'] = df_features['YearRemodAdd'].astype('str')
            
            # 7. Neighborhood Price Category (based on median prices)
            if all(col in df_features.columns for col in ['Neighborhood', 'SalePrice']):
                # Calculate median price by neighborhood
                neighborhood_prices = df_features.groupby('Neighborhood')['SalePrice'].median()
                
                # Create neighborhood price categories
                price_quartiles = neighborhood_prices.quantile([0.25, 0.5, 0.75])
                
                def get_price_category(neighborhood):
                    median_price = neighborhood_prices[neighborhood]
                    if median_price <= price_quartiles[0.25]:
                        return 'Low'
                    elif median_price <= price_quartiles[0.5]:
                        return 'Medium-Low'
                    elif median_price <= price_quartiles[0.75]:
                        return 'Medium-High'
                    else:
                        return 'High'
                
                df_features['NeighborhoodCategory'] = df_features['Neighborhood'].apply(get_price_category)
            
            logger.info(f"Created features for Ames Housing dataset. New shape: {df_features.shape}")
            return df_features
        except Exception as e:
            logger.error(f"Error creating features for Ames Housing: {e}")
            raise
    
    def create_features_california(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from the California Housing dataset.
        
        Args:
            df: Cleaned California Housing dataset
            
        Returns:
            Dataframe with engineered features
        """
        try:
            # Make a copy to avoid modifying the original
            df_features = df.copy()
            
            # Create new features
            
            # 1. Total Rooms per Household
            if all(col in df_features.columns for col in ['AveRooms', 'HouseHolds']):
                df_features['TotalRooms'] = df_features['AveRooms'] * df_features['HouseHolds']
            
            # 2. Total Bedrooms per Household
            if all(col in df_features.columns for col in ['AveBedrms', 'HouseHolds']):
                df_features['TotalBedrooms'] = df_features['AveBedrms'] * df_features['HouseHolds']
            
            # 3. Population per Household
            if all(col in df_features.columns for col in ['Population', 'HouseHolds']):
                df_features['PopulationPerHousehold'] = df_features['Population'] / df_features['HouseHolds']
                
            # 4. Bedrooms per Room ratio
            if all(col in df_features.columns for col in ['AveBedrms', 'AveRooms']):
                df_features['BedroomRatio'] = df_features['AveBedrms'] / df_features['AveRooms']
            
            # 5. Income per Population ratio (per capita income)
            if all(col in df_features.columns for col in ['MedInc', 'Population']):
                df_features['IncomePerCapita'] = df_features['MedInc'] / df_features['Population']
            
            # 6. Distance to major city centers (if coordinates available)
            if all(col in df_features.columns for col in ['Latitude', 'Longitude']):
                # Define major California city centers (approximate coordinates)
                sf_coords = (37.7749, -122.4194)  # San Francisco
                la_coords = (34.0522, -118.2437)  # Los Angeles
                sd_coords = (32.7157, -117.1611)  # San Diego
                
                # Calculate Haversine distance (km) to each city
                from math import radians, cos, sin, asin, sqrt
                
                def haversine(lat1, lon1, lat2, lon2):
                    # Convert degrees to radians
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                    
                    # Haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    r = 6371  # Radius of Earth in kilometers
                    return c * r
                
                # Distance to San Francisco
                df_features['DistToSF'] = df_features.apply(
                    lambda row: haversine(row['Latitude'], row['Longitude'], 
                                         sf_coords[0], sf_coords[1]), 
                    axis=1
                )
                
                # Distance to Los Angeles
                df_features['DistToLA'] = df_features.apply(
                    lambda row: haversine(row['Latitude'], row['Longitude'], 
                                         la_coords[0], la_coords[1]), 
                    axis=1
                )
                
                # Distance to San Diego
                df_features['DistToSD'] = df_features.apply(
                    lambda row: haversine(row['Latitude'], row['Longitude'], 
                                         sd_coords[0], sd_coords[1]), 
                    axis=1
                )
                
                # Minimum distance to any major city
                df_features['MinDistToCity'] = df_features[['DistToSF', 'DistToLA', 'DistToSD']].min(axis=1)
            
            # 7. Binned features
            # Bin median income
            if 'MedInc' in df_features.columns:
                df_features['MedIncBin'] = pd.qcut(df_features['MedInc'], 5, labels=False)
            
            # Bin house age
            if 'HouseAge' in df_features.columns:
                df_features['HouseAgeBin'] = pd.qcut(df_features['HouseAge'], 4, labels=False)
            
            logger.info(f"Created features for California Housing dataset. New shape: {df_features.shape}")
            return df_features
        except Exception as e:
            logger.error(f"Error creating features for California Housing: {e}")
            raise
    
    def preprocess_ames_housing(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
        """
        Preprocess the Ames Housing dataset for modeling.
        
        Args:
            df: Feature-engineered Ames Housing dataset
            target_col: Name of the target column
            
        Returns:
            X: Features dataframe
            y: Target series
            preprocessor: Fitted column transformer for preprocessing
        """
        try:
            # Separate features and target
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe")
            
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Log transform the target variable if it's SalePrice
            if target_col == 'SalePrice':
                df_copy[target_col] = np.log1p(df_copy[target_col])
                logger.info(f"Log-transformed {target_col}")
            
            # Split X and y
            y = df_copy[target_col]
            X = df_copy.drop(columns=[target_col])
            
            # Identify column types
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
            
            # Create preprocessing transformers
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )
            
            logger.info(f"Created preprocessor with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
            
            return X, y, preprocessor
        except Exception as e:
            logger.error(f"Error preprocessing Ames Housing data: {e}")
            raise
    
    def preprocess_california_housing(self, df: pd.DataFrame, target_col: str = 'PRICE') -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
        """
        Preprocess the California Housing dataset for modeling.
        
        Args:
            df: Feature-engineered California Housing dataset
            target_col: Name of the target column
            
        Returns:
            X: Features dataframe
            y: Target series
            preprocessor: Fitted column transformer for preprocessing
        """
        try:
            # Separate features and target
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe")
            
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Split X and y
            y = df_copy[target_col]
            X = df_copy.drop(columns=[target_col])
            
            # Identify column types
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
            
            # Create preprocessing transformers
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ],
                remainder='passthrough'
            )
            
            logger.info(f"Created preprocessor with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
            
            return X, y, preprocessor
        except Exception as e:
            logger.error(f"Error preprocessing California Housing data: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save processed dataframe to the processed data directory."""
        try:
            output_path = os.path.join(self.processed_data_path, filename)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def save_features(self, df: pd.DataFrame, filename: str) -> str:
        """Save feature dataframe to the features directory."""
        try:
            output_path = os.path.join(self.features_path, filename)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved features to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise
    
    def save_preprocessor(self, preprocessor: ColumnTransformer, filename: str) -> str:
        """Save the fitted preprocessor to disk."""
        try:
            output_path = os.path.join(self.processed_data_path, filename)
            with open(output_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            logger.info(f"Saved preprocessor to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            raise

    def process_ames_housing_pipeline(self, input_filename: str, output_filename_base: str) -> Dict[str, str]:
        """
        Run the full processing pipeline for Ames Housing dataset.
        
        Args:
            input_filename: Filename of the raw data
            output_filename_base: Base name for output files
            
        Returns:
            Dictionary with paths to output files
        """
        try:
            # Load data
            df_raw = self.load_data(input_filename)
            
            # Clean data
            df_cleaned = self.clean_ames_housing_data(df_raw)
            cleaned_path = self.save_processed_data(df_cleaned, f"cleaned_{output_filename_base}")
            
            # Create features
            df_features = self.create_features_ames(df_cleaned)
            features_path = self.save_features(df_features, f"features_{output_filename_base}")
            
            # Preprocess for modeling
            X, y, preprocessor = self.preprocess_ames_housing(df_features)
            
            # Save preprocessor
            preprocessor_path = self.save_preprocessor(preprocessor, f"preprocessor_{output_filename_base}.pkl")
            
            # Save X and y separately for model training
            X_path = self.save_features(X, f"X_{output_filename_base}")
            y_df = pd.DataFrame({y.name: y})
            y_path = self.save_features(y_df, f"y_{output_filename_base}")
            
            return {
                "cleaned_data": cleaned_path,
                "features": features_path,
                "preprocessor": preprocessor_path,
                "X": X_path,
                "y": y_path
            }
        except Exception as e:
            logger.error(f"Error in Ames Housing processing pipeline: {e}")
            raise

    def process_california_housing_pipeline(self, input_filename: str, output_filename_base: str) -> Dict[str, str]:
        """
        Run the full processing pipeline for California Housing dataset.
        
        Args:
            input_filename: Filename of the raw data
            output_filename_base: Base name for output files
            
        Returns:
            Dictionary with paths to output files
        """
        try:
            # Load data
            df_raw = self.load_data(input_filename)
            
            # Clean data
            df_cleaned = self.clean_california_housing_data(df_raw)
            cleaned_path = self.save_processed_data(df_cleaned, f"cleaned_{output_filename_base}")
            
            # Create features
            df_features = self.create_features_california(df_cleaned)
            features_path = self.save_features(df_features, f"features_{output_filename_base}")
            
            # Preprocess for modeling
            X, y, preprocessor = self.preprocess_california_housing(df_features)
            
            # Save preprocessor
            preprocessor_path = self.save_preprocessor(preprocessor, f"preprocessor_{output_filename_base}.pkl")
            
            # Save X and y separately for model training
            X_path = self.save_features(X, f"X_{output_filename_base}")
            y_df = pd.DataFrame({'PRICE': y})
            y_path = self.save_features(y_df, f"y_{output_filename_base}")
            
            return {
                "cleaned_data": cleaned_path,
                "features": features_path,
                "preprocessor": preprocessor_path,
                "X": X_path,
                "y": y_path
            }
        except Exception as e:
            logger.error(f"Error in California Housing processing pipeline: {e}")
            raise