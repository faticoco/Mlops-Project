import os
import sys
import numpy as np
import mlflow
import pandas as pd

def load_model(model_name="housing_predict"):
    """
    Load the model from MLflow model registry
    
    Args:
        model_name: Name of the registered model
    
    Returns:
        Loaded model
    """
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        
        model_uri = f"models:/{model_name}/Production"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{model_name}' from MLflow registry") 
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            model_uri = f"models:/{model_name}/latest"
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded latest version of model '{model_name}'")
            return loaded_model
        except Exception as e2:
            print(f"Failed to load latest version: {e2}")
            sys.exit(1)

def prepare_input_data(input_data, feature_names=None):
    """
    Prepare input data for prediction
    
    Args:
        input_data: Input data as a list of lists or numpy array
        feature_names: List of feature names (optional)
    
    Returns:
        Prepared input data (DataFrame or numpy array)
    """
    if isinstance(input_data, list):
        if feature_names:
            return pd.DataFrame(input_data, columns=feature_names)
        else:
            return np.array(input_data)
    elif isinstance(input_data, np.ndarray):
        if feature_names and input_data.ndim == 2:
            return pd.DataFrame(input_data, columns=feature_names)
        else:
            return input_data
    else:
        return input_data

def make_predictions(model, input_data):
    """
    Make predictions using the loaded model
    
    Args:
        model: Loaded MLflow model
        input_data: Prepared input data
        
    Returns:
        Predictions
    """
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def main():
    # Load the model from MLflow model registry
    model = load_model()
    
    # Define feature names for Ames housing dataset (example)
    feature_names = [
        "LotArea", "OverallQual", "OverallCond", "YearBuilt", 
        "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
        "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
        "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
        "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF",
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
        "PoolArea", "MiscVal", "MoSold", "YrSold"
    ]
    
    # Create sample data for demonstration
    # These are dummy values for the Ames Housing dataset features
    sample_data = [
        # Sample 1: High-end house
        [10000, 9, 7, 2015, 2015, 1200, 0, 300, 1500, 1800, 1200, 0, 
         3000, 1, 1, 2, 1, 4, 1, 10, 2, 3, 650, 200, 100, 0, 0, 0, 0, 0, 6, 2022],
         
        # Sample 2: Medium-range house
        [8000, 7, 5, 2000, 2010, 900, 0, 400, 1300, 1500, 900, 0, 
         2400, 1, 0, 2, 0, 3, 1, 8, 1, 2, 440, 100, 50, 0, 0, 0, 0, 0, 4, 2022],
         
        # Sample 3: Lower-end house
        [6000, 5, 4, 1980, 1995, 600, 0, 300, 900, 1100, 0, 0, 
         1100, 0, 0, 1, 0, 2, 1, 5, 0, 1, 250, 0, 20, 0, 0, 0, 0, 0, 5, 2022]
    ]
    
    # Prepare the input data
    input_data = prepare_input_data(sample_data, feature_names)
    
    # Make predictions
    predictions = make_predictions(model, input_data)
    
    # Display the results
    if predictions is not None:
        print("\nHousing Price Predictions:")
        print("-" * 60)
        
        for i, (sample, prediction) in enumerate(zip(sample_data, predictions)):
            print(f"Sample {i+1}:")
            # Print just a few key features to keep output manageable
            print(f"  Lot Area: {sample[0]}")
            print(f"  Overall Quality: {sample[1]}/10")
            print(f"  Year Built: {sample[3]}")
            print(f"  Total Living Area: {sample[12]} sq ft")
            print(f"  Bedrooms: {sample[17]}")
            print(f"  Bathrooms: {sample[15]} full, {sample[16]} half")
            print(f"  Predicted Price: ${prediction:,.2f}")
            print("-" * 60)

if __name__ == "__main__":
    main()