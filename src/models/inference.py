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

def get_default_values():
    """
    Provides default values for all features required by the model
    
    Returns:
        Dictionary of default values for all features
    """
    # Numerical defaults - using median values as reasonable defaults
    numerical_defaults = {
        "Id": 0,  # Will be auto-incremented
        "MSSubClass": 50,  # 1950s average quality homes
        "LotFrontage": 70.0,  # median lot frontage
        "LotArea": 9000,  # median lot area
        "OverallQual": 6,  # average quality
        "OverallCond": 5,  # average condition
        "YearBuilt": 1980,  # median year
        "YearRemodAdd": 1980,  # no remodel by default
        "MasVnrArea": 0.0,  # no masonry veneer
        "BsmtFinSF1": 400,  # some finished basement
        "BsmtFinSF2": 0,  # no secondary finished basement
        "BsmtUnfSF": 500,  # some unfinished basement
        "TotalBsmtSF": 900,  # total basement area
        "1stFlrSF": 1200,  # first floor square feet
        "2ndFlrSF": 0,  # no second floor
        "LowQualFinSF": 0,  # no low quality finish
        "GrLivArea": 1500,  # above ground living area
        "BsmtFullBath": 0,  # no basement bathroom
        "BsmtHalfBath": 0,  # no basement half bath
        "FullBath": 2,  # typical full baths
        "HalfBath": 0,  # no half baths
        "BedroomAbvGr": 3,  # typical bedrooms
        "KitchenAbvGr": 1,  # typical kitchen
        "TotRmsAbvGrd": 6,  # total rooms
        "Fireplaces": 1,  # one fireplace
        "GarageYrBlt": 1980.0,  # garage built with house
        "GarageCars": 2,  # 2 car garage
        "GarageArea": 480,  # 2 car garage size
        "WoodDeckSF": 0,  # no wood deck
        "OpenPorchSF": 40,  # small porch
        "EnclosedPorch": 0,  # no enclosed porch
        "3SsnPorch": 0,  # no 3 season porch
        "ScreenPorch": 0,  # no screen porch
        "PoolArea": 0,  # no pool
        "MiscVal": 0,  # no miscellaneous features
        "MoSold": 6,  # June (median)
        "YrSold": 2010,  # median year
        "TotalSF": 2400,  # total square footage
        "TotalBaths": 2.0,  # total bathrooms
    }
    
    # Derived fields
    numerical_defaults["HasPool"] = 1 if numerical_defaults["PoolArea"] > 0 else 0
    numerical_defaults["HasGarage"] = 1 if numerical_defaults["GarageCars"] > 0 else 0
    numerical_defaults["Remodeled"] = 1 if numerical_defaults["YearRemodAdd"] > numerical_defaults["YearBuilt"] else 0
    
    # Create all categorical features as False by default
    categorical_features = [
        # MSZoning features
        "MSZoning_FV", "MSZoning_RH", "MSZoning_RL", "MSZoning_RM",
        # Street features
        "Street_Pave",
        # LotShape features
        "LotShape_IR2", "LotShape_IR3", "LotShape_Reg",
        # LandContour features
        "LandContour_HLS", "LandContour_Low", "LandContour_Lvl",
        # Utilities features
        "Utilities_NoSeWa",
        # LotConfig features
        "LotConfig_CulDSac", "LotConfig_FR2", "LotConfig_FR3", "LotConfig_Inside",
        # LandSlope features
        "LandSlope_Mod", "LandSlope_Sev",
        # Neighborhood features
        "Neighborhood_Blueste", "Neighborhood_BrDale", "Neighborhood_BrkSide",
        "Neighborhood_ClearCr", "Neighborhood_CollgCr", "Neighborhood_Crawfor",
        "Neighborhood_Edwards", "Neighborhood_Gilbert", "Neighborhood_IDOTRR",
        "Neighborhood_MeadowV", "Neighborhood_Mitchel", "Neighborhood_NAmes",
        "Neighborhood_NPkVill", "Neighborhood_NWAmes", "Neighborhood_NoRidge",
        "Neighborhood_NridgHt", "Neighborhood_OldTown", "Neighborhood_SWISU",
        "Neighborhood_Sawyer", "Neighborhood_SawyerW", "Neighborhood_Somerst",
        "Neighborhood_StoneBr", "Neighborhood_Timber", "Neighborhood_Veenker",
        # Condition1 features
        "Condition1_Feedr", "Condition1_Norm", "Condition1_PosA", "Condition1_PosN",
        "Condition1_RRAe", "Condition1_RRAn", "Condition1_RRNe", "Condition1_RRNn",
        # Condition2 features
        "Condition2_Feedr", "Condition2_Norm", "Condition2_PosA", "Condition2_PosN",
        "Condition2_RRAe", "Condition2_RRAn", "Condition2_RRNn",
        # BldgType features
        "BldgType_2fmCon", "BldgType_Duplex", "BldgType_Twnhs", "BldgType_TwnhsE",
        # HouseStyle features 
        "HouseStyle_1.5Unf", "HouseStyle_1Story", "HouseStyle_2.5Fin", "HouseStyle_2.5Unf",
        "HouseStyle_2Story", "HouseStyle_SFoyer", "HouseStyle_SLvl",
        # RoofStyle features
        "RoofStyle_Gable", "RoofStyle_Gambrel", "RoofStyle_Hip", "RoofStyle_Mansard", "RoofStyle_Shed",
        # RoofMatl features
        "RoofMatl_CompShg", "RoofMatl_Membran", "RoofMatl_Metal", "RoofMatl_Roll",
        "RoofMatl_Tar&Grv", "RoofMatl_WdShake", "RoofMatl_WdShngl",
        # ... All other categorical features ...
    ]
    
    # Continue with all the other categorical features from the error message
    other_categorical_features = [
        # Exterior1st features
        "Exterior1st_AsphShn", "Exterior1st_BrkComm", "Exterior1st_BrkFace", "Exterior1st_CBlock",
        "Exterior1st_CemntBd", "Exterior1st_HdBoard", "Exterior1st_ImStucc", "Exterior1st_MetalSd",
        "Exterior1st_Plywood", "Exterior1st_Stone", "Exterior1st_Stucco", "Exterior1st_VinylSd",
        "Exterior1st_Wd Sdng", "Exterior1st_WdShing",
        # Exterior2nd features
        "Exterior2nd_AsphShn", "Exterior2nd_Brk Cmn", "Exterior2nd_BrkFace", "Exterior2nd_CBlock",
        "Exterior2nd_CmentBd", "Exterior2nd_HdBoard", "Exterior2nd_ImStucc", "Exterior2nd_MetalSd",
        "Exterior2nd_Other", "Exterior2nd_Plywood", "Exterior2nd_Stone", "Exterior2nd_Stucco",
        "Exterior2nd_VinylSd", "Exterior2nd_Wd Sdng", "Exterior2nd_Wd Shng",
        # MasVnrType features
        "MasVnrType_BrkFace", "MasVnrType_Stone",
        # ExterQual features
        "ExterQual_Fa", "ExterQual_Gd", "ExterQual_TA",
        # ExterCond features
        "ExterCond_Fa", "ExterCond_Gd", "ExterCond_Po", "ExterCond_TA",
        # Foundation features
        "Foundation_CBlock", "Foundation_PConc", "Foundation_Slab", "Foundation_Stone", "Foundation_Wood",
        # BsmtQual features
        "BsmtQual_Fa", "BsmtQual_Gd", "BsmtQual_TA",
        # BsmtCond features
        "BsmtCond_Gd", "BsmtCond_Po", "BsmtCond_TA",
        # BsmtExposure features
        "BsmtExposure_Gd", "BsmtExposure_Mn", "BsmtExposure_No",
        # BsmtFinType1 features
        "BsmtFinType1_BLQ", "BsmtFinType1_GLQ", "BsmtFinType1_LwQ", "BsmtFinType1_Rec", "BsmtFinType1_Unf",
        # BsmtFinType2 features
        "BsmtFinType2_BLQ", "BsmtFinType2_GLQ", "BsmtFinType2_LwQ", "BsmtFinType2_Rec", "BsmtFinType2_Unf",
        # Heating features
        "Heating_GasA", "Heating_GasW", "Heating_Grav", "Heating_OthW", "Heating_Wall",
        # HeatingQC features
        "HeatingQC_Fa", "HeatingQC_Gd", "HeatingQC_Po", "HeatingQC_TA",
        # CentralAir features
        "CentralAir_Y",
        # Electrical features
        "Electrical_FuseF", "Electrical_FuseP", "Electrical_Mix", "Electrical_SBrkr",
        # KitchenQual features
        "KitchenQual_Fa", "KitchenQual_Gd", "KitchenQual_TA",
        # Functional features
        "Functional_Maj2", "Functional_Min1", "Functional_Min2", "Functional_Mod", "Functional_Sev", "Functional_Typ",
        # FireplaceQu features
        "FireplaceQu_Fa", "FireplaceQu_Gd", "FireplaceQu_Po", "FireplaceQu_TA",
        # GarageType features
        "GarageType_Attchd", "GarageType_Basment", "GarageType_BuiltIn", "GarageType_CarPort", "GarageType_Detchd",
        # GarageFinish features
        "GarageFinish_RFn", "GarageFinish_Unf",
        # GarageQual features
        "GarageQual_Fa", "GarageQual_Gd", "GarageQual_Po", "GarageQual_TA",
        # GarageCond features
        "GarageCond_Fa", "GarageCond_Gd", "GarageCond_Po", "GarageCond_TA",
        # PavedDrive features
        "PavedDrive_P", "PavedDrive_Y",
        # SaleType features
        "SaleType_CWD", "SaleType_Con", "SaleType_ConLD", "SaleType_ConLI", "SaleType_ConLw",
        "SaleType_New", "SaleType_Oth", "SaleType_WD",
        # SaleCondition features
        "SaleCondition_AdjLand", "SaleCondition_Alloca", "SaleCondition_Family", "SaleCondition_Normal", "SaleCondition_Partial",
        # NeighborhoodCategory features
        "NeighborhoodCategory_Low", "NeighborhoodCategory_Medium-High", "NeighborhoodCategory_Medium-Low"
    ]
    
    categorical_features.extend(other_categorical_features)
    
    # Initialize all categorical features as False by default
    categorical_defaults = {feature: False for feature in categorical_features}
    
    # Set a few sensible default category values to True
    # These represent typical house characteristics
    sensible_defaults = [
        "MSZoning_RL",  # Residential Low Density
        "Street_Pave",  # Paved Street
        "LotShape_Reg",  # Regular Lot Shape
        "LandContour_Lvl",  # Level Land Contour
        "LotConfig_Inside",  # Inside Lot
        "Neighborhood_NAmes",  # North Ames (common neighborhood)
        "Condition1_Norm",  # Normal Condition
        "Condition2_Norm",  # Normal Condition
        "HouseStyle_1Story",  # 1 Story House
        "RoofStyle_Gable",  # Gable Roof
        "RoofMatl_CompShg",  # Standard Composite Shingle Roof
        "Exterior1st_VinylSd",  # Vinyl Siding
        "Exterior2nd_VinylSd",  # Vinyl Siding
        "Foundation_PConc",  # Poured Concrete Foundation
        "BsmtQual_TA",  # Typical Average Basement Quality
        "BsmtCond_TA",  # Typical Average Basement Condition
        "BsmtExposure_No",  # No Basement Exposure
        "BsmtFinType1_Rec",  # Rec Room Basement Finish
        "BsmtFinType2_Unf",  # Unfinished Basement
        "Heating_GasA",  # Gas Forced Warm Air Heating
        "HeatingQC_TA",  # Typical Average Heating Quality
        "CentralAir_Y",  # Central Air Conditioning
        "Electrical_SBrkr",  # Standard Circuit Breakers
        "KitchenQual_TA",  # Typical Average Kitchen Quality
        "Functional_Typ",  # Typical Functionality
        "GarageType_Attchd",  # Attached Garage
        "GarageFinish_Unf",  # Unfinished Garage
        "GarageQual_TA",  # Typical Average Garage Quality
        "GarageCond_TA",  # Typical Average Garage Condition
        "PavedDrive_Y",  # Paved Driveway
        "SaleType_WD",  # Warranty Deed - Conventional
        "SaleCondition_Normal",  # Normal Sale Condition
        "NeighborhoodCategory_Medium-Low"  # Medium to Low Neighborhood Category
    ]
    
    # Set the default categories to True
    for feature in sensible_defaults:
        categorical_defaults[feature] = True
    
    # Combine numerical and categorical defaults
    all_defaults = {**numerical_defaults, **categorical_defaults}
    
    return all_defaults

def prepare_input_data_with_defaults(user_data, default_values=None):
    """
    Prepares input data with defaults for prediction
    
    Args:
        user_data: Input data provided by user as a list of dictionaries or pandas DataFrame
        default_values: Dictionary of default values for all features
        
    Returns:
        Prepared input data as pandas DataFrame with all required features
    """
    if default_values is None:
        default_values = get_default_values()
    
    # Convert user data to DataFrame if it's not already
    if isinstance(user_data, list):
        if all(isinstance(item, dict) for item in user_data):
            user_df = pd.DataFrame(user_data)
        else:
            # Assume it's a list of lists with the feature_names from the original code
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
            user_df = pd.DataFrame(user_data, columns=feature_names)
    elif isinstance(user_data, pd.DataFrame):
        user_df = user_data.copy()
    else:
        raise ValueError("User data must be a list of dictionaries, list of lists, or pandas DataFrame")
    
    # Create a DataFrame with default values for all required features
    rows = len(user_df)
    default_df = pd.DataFrame([default_values] * rows)
    
    # Auto-increment Id if it's in the default values
    if "Id" in default_df.columns:
        default_df["Id"] = list(range(1, rows + 1))
    
    # Update with user-provided values
    for column in user_df.columns:
        if column in default_df.columns:
            default_df[column] = user_df[column]
    
    # Calculate derived fields based on updated values
    default_df["TotalSF"] = (default_df["1stFlrSF"] + default_df["2ndFlrSF"] + default_df["TotalBsmtSF"]).astype(int)
    default_df["TotalBaths"] = (default_df["FullBath"] + (0.5 * default_df["HalfBath"]) + \
                              default_df["BsmtFullBath"] + (0.5 * default_df["BsmtHalfBath"])).astype(int)
    default_df["HasPool"] = (default_df["PoolArea"] > 0).astype(int)
    default_df["HasGarage"] = (default_df["GarageCars"] > 0).astype(int)
    default_df["Remodeled"] = (default_df["YearRemodAdd"] > default_df["YearBuilt"]).astype(int)
    
    # Ensure all numeric columns are of the correct type
    numeric_columns = {
        'Id': 'int64',
        'MSSubClass': 'int64',
        'LotFrontage': 'float64',
        'LotArea': 'int64',
        'OverallQual': 'int64',
        'OverallCond': 'int64',
        'YearBuilt': 'int64',
        'YearRemodAdd': 'int64',
        'MasVnrArea': 'float64',
        'BsmtFinSF1': 'int64',
        'BsmtFinSF2': 'int64',
        'BsmtUnfSF': 'int64',
        'TotalBsmtSF': 'int64',
        '1stFlrSF': 'int64',
        '2ndFlrSF': 'int64',
        'LowQualFinSF': 'int64',
        'GrLivArea': 'int64',
        'BsmtFullBath': 'int64',
        'BsmtHalfBath': 'int64',
        'FullBath': 'int64',
        'HalfBath': 'int64',
        'BedroomAbvGr': 'int64',
        'KitchenAbvGr': 'int64',
        'TotRmsAbvGrd': 'int64',
        'Fireplaces': 'int64',
        'GarageYrBlt': 'float64',
        'GarageCars': 'int64',
        'GarageArea': 'int64',
        'WoodDeckSF': 'int64',
        'OpenPorchSF': 'int64',
        'EnclosedPorch': 'int64',
        '3SsnPorch': 'int64',
        'ScreenPorch': 'int64',
        'PoolArea': 'int64',
        'MiscVal': 'int64',
        'MoSold': 'int64',
        'YrSold': 'int64',
        'TotalSF': 'int64',
        'TotalBaths': 'int64',
        'HasPool': 'int64',
        'HasGarage': 'int64',
        'Remodeled': 'int64'
    }
    
    # Convert numeric columns to their correct types
    for col, dtype in numeric_columns.items():
        if col in default_df.columns:
            default_df[col] = default_df[col].astype(dtype)
    
    # Ensure all boolean columns are of the correct type
    for col in default_df.columns:
        if col.startswith(("MSZoning_", "Street_", "LotShape_")) or \
           any(col.startswith(prefix) for prefix in [
               "LandContour_", "Utilities_", "LotConfig_", "LandSlope_", "Neighborhood_",
               "Condition1_", "Condition2_", "BldgType_", "HouseStyle_", "RoofStyle_",
               "RoofMatl_", "Exterior1st_", "Exterior2nd_", "MasVnrType_", "ExterQual_",
               "ExterCond_", "Foundation_", "BsmtQual_", "BsmtCond_", "BsmtExposure_",
               "BsmtFinType1_", "BsmtFinType2_", "Heating_", "HeatingQC_", "CentralAir_",
               "Electrical_", "KitchenQual_", "Functional_", "FireplaceQu_", "GarageType_",
               "GarageFinish_", "GarageQual_", "GarageCond_", "PavedDrive_", "SaleType_",
               "SaleCondition_", "NeighborhoodCategory_"
           ]):
            default_df[col] = default_df[col].astype(bool)
    
    return default_df

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
    
    # Sample user data - only providing a few fields
    user_data = [
        # User only specifies these features, the rest will use defaults
        {"LotArea": 10000, "OverallQual": 9, "OverallCond": 7, "YearBuilt": 2015, "YearRemodAdd": 2015},
        {"LotArea": 8000, "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2000, "YearRemodAdd": 2010},
        {"LotArea": 6000, "OverallQual": 5, "OverallCond": 4, "YearBuilt": 1980, "YearRemodAdd": 1995}
    ]
    
    # Or alternatively, if user data is provided as a list of lists with known feature names:
    feature_names = [
        "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"
    ]
    user_data_list = [
        [10000, 9, 7, 2015, 2015],
        [8000, 7, 5, 2000, 2010],
        [6000, 5, 4, 1980, 1995]
    ]
    user_df = pd.DataFrame(user_data_list, columns=feature_names)
    
    # Prepare the input data with defaults
    input_data = prepare_input_data_with_defaults(user_data)
    
    # Make predictions
    predictions = make_predictions(model, input_data)
    
    # Display the results
    if predictions is not None:
        print("\nHousing Price Predictions:")
        print("-" * 60)
        
        for i, (user_entry, prediction) in enumerate(zip(user_data, predictions)):
            print(f"Sample {i+1}:")
            for key, value in user_entry.items():
                print(f"  {key}: {value}")
            print(f"  Predicted Price: ${np.expm1(prediction):,.2f}")
            print("-" * 60)

if __name__ == "__main__":
    main()